from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from controlflow.core.resources import GpuSemaphore
from controlflow.core.state import ProjectPaths, canonical_json, sha256_file, utc_now
from controlflow.data.splits import load_split
from controlflow.eval.metrics import classification_metrics
from controlflow.modeling.supervised import CATEGORICAL, LABELS, NUMERIC


class MLP(nn.Module):
    def __init__(self, dimensions: int, classes: int = 4) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dimensions, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.layers(features))


def run_mlp_validation(cases_path: Path, seeds: tuple[int, ...] = (17,)) -> pd.DataFrame:
    if not torch.cuda.is_available():
        raise RuntimeError("eligible PyTorch MLP workload refused CPU fallback")
    frame = pd.read_parquet(cases_path)
    train = frame[frame["case_id"].isin(set(load_split("train", phase="P11")))]
    validation = frame[frame["case_id"].isin(set(load_split("validation", phase="P11")))]
    transformer = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )
    train_x = transformer.fit_transform(train).astype("float32")
    val_x = transformer.transform(validation).astype("float32")
    train_y = pd.Categorical(train["severity"], categories=LABELS).codes.astype("int64")
    val_y = pd.Categorical(validation["severity"], categories=LABELS).codes.astype("int64")
    records: list[dict[str, Any]] = []
    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
            batch_size=256,
            shuffle=True,
            generator=generator,
            pin_memory=True,
            num_workers=0,
        )
        started = time.perf_counter()
        with GpuSemaphore():
            model = MLP(train_x.shape[1]).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            with torch.amp.autocast("cuda", dtype=torch.float16):
                model.train()
                for _ in range(40):
                    for features, labels in loader:
                        features = features.cuda(non_blocking=True)
                        labels = labels.cuda(non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)
                        loss = criterion(model(features), labels)
                        loss.backward()
                        optimizer.step()
                model.eval()
                with torch.inference_mode():
                    logits = model(torch.from_numpy(val_x).cuda())
                    probabilities = torch.softmax(logits, dim=1).float().cpu().numpy()
            device = str(next(model.parameters()).device)
            del model, optimizer
            torch.cuda.empty_cache()
        if device != "cuda:0":
            raise RuntimeError(f"MLP device attestation failed: {device}")
        metrics = classification_metrics(val_y, probabilities)
        config = {"model": "M5_pytorch_mlp", "seed": seed, "epochs": 40, "device": "cuda"}
        records.append(
            {
                "experiment_id": f"risk-M5_pytorch_mlp-s{seed}",
                "config_hash": hashlib.sha256(canonical_json(config)).hexdigest(),
                "dataset_hash": sha256_file(cases_path),
                "split_identifier": "validation",
                "seed": seed,
                "hardware_runtime": f"{platform.system()}-{platform.machine()}-cuda:{torch.cuda.get_device_name(0)}",
                "timestamp": utc_now(),
                "status": "ok",
                "runtime_seconds": time.perf_counter() - started,
                "metrics": json.dumps(metrics, sort_keys=True),
                "details": json.dumps(
                    {
                        "torch": torch.__version__,
                        "compiled_cuda": torch.version.cuda,
                        "device": device,
                    },
                    sort_keys=True,
                ),
            }
        )
    result = pd.DataFrame(records)
    target = ProjectPaths.discover().root / "results" / "ml_models_mlp.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(target, index=False)
    return result


class Autoencoder(nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        latent = max(2, dimensions // 2)
        self.encoder = nn.Sequential(nn.Linear(dimensions, 16), nn.ReLU(), nn.Linear(16, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, dimensions))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.decoder(self.encoder(features)))
