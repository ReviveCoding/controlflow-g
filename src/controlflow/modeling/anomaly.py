from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from controlflow.core.resources import GpuSemaphore
from controlflow.core.state import PhaseRun, ProjectPaths, canonical_json, sha256_file, utc_now
from controlflow.data.splits import load_split
from controlflow.modeling.torch_models import Autoencoder

FEATURES = ["amount", "repeat_count", "historical_failures", "data_sensitivity"]


def _metrics(
    truth: np.ndarray[Any, Any],
    score: np.ndarray[Any, Any],
    predicted: np.ndarray[Any, Any],
    novel: np.ndarray[Any, Any],
) -> dict[str, float]:
    normal = ~truth
    return {
        "pr_auc": float(average_precision_score(truth, score)),
        "anomaly_recall": float(recall_score(truth, predicted, zero_division=0)),
        "false_positive_rate": float(predicted[normal].mean()) if normal.any() else 0.0,
        "novel_case_recall": float(predicted[novel].mean()) if novel.any() else float("nan"),
    }


def run_anomaly_validation(cases_path: Path, seed: int = 17) -> pd.DataFrame:
    frame = pd.read_parquet(cases_path)
    train_ids = set(load_split("train", phase="P13"))
    ids = set(load_split("validation", phase="P13") + load_split("ood", phase="P13"))
    train = frame[frame["case_id"].isin(train_ids) & ~frame["case_type"].isin(["rare", "critical", "adversarial"])]
    selected = frame[frame["case_id"].isin(ids)].copy()
    scaler = StandardScaler().fit(train[FEATURES])
    train_x = scaler.transform(train[FEATURES])
    x = scaler.transform(selected[FEATURES])
    truth = selected["case_type"].isin(["rare", "critical", "adversarial"]).to_numpy()
    novel = selected["is_ood"].to_numpy()
    contamination = 0.05
    models = {
        "U0_statistical": None,
        "U1_isolation_forest": IsolationForest(contamination=contamination, random_state=seed, n_jobs=1),
        "U2_local_outlier_factor": LocalOutlierFactor(
            n_neighbors=min(35, len(train_x) - 1), contamination=contamination, novelty=True
        ),
    }
    records: list[dict[str, Any]] = []
    for name, model in models.items():
        started = time.perf_counter()
        if model is None:
            score = np.abs(x).max(axis=1)
            threshold = np.quantile(np.abs(train_x).max(axis=1), 1 - contamination)
            predicted = score >= threshold
        else:
            model.fit(train_x)
            labels = model.predict(x)
            score = -model.score_samples(x)
            predicted = labels == -1
        config = {"model": name, "seed": seed, "contamination": contamination}
        for split_name, mask in {
            "validation": ~selected.is_ood.to_numpy(),
            "ood": selected.is_ood.to_numpy(),
        }.items():
            metrics = _metrics(truth[mask], score[mask], predicted[mask], novel[mask])
            records.append(
                {
                    "experiment_id": f"anomaly-{name}-{split_name}-s{seed}",
                    "config_hash": hashlib.sha256(canonical_json(config)).hexdigest(),
                    "dataset_hash": sha256_file(cases_path),
                    "split_identifier": split_name,
                    "seed": seed,
                    "hardware_runtime": f"{platform.system()}-{platform.machine()}",
                    "timestamp": utc_now(),
                    "status": "ok",
                    "runtime_seconds": time.perf_counter() - started,
                    "metrics": json.dumps(metrics, sort_keys=True),
                }
            )
    result = pd.DataFrame(records)
    target = ProjectPaths.discover().root / "results" / "anomaly.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(target, index=False)
    return result


def run_autoencoder_validation(cases_path: Path, seed: int = 17) -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("autoencoder experiment refused CPU fallback")
    frame = pd.read_parquet(cases_path)
    train_ids = set(load_split("train", phase="P13"))
    evaluation_ids = set(load_split("validation", phase="P13") + load_split("ood", phase="P13"))
    train = frame[frame.case_id.isin(train_ids) & ~frame.case_type.isin(["rare", "critical", "adversarial"])]
    evaluation = frame[frame.case_id.isin(evaluation_ids)]
    scaler = StandardScaler().fit(train[FEATURES])
    train_x = torch.tensor(scaler.transform(train[FEATURES]), dtype=torch.float32)
    evaluation_x = torch.tensor(scaler.transform(evaluation[FEATURES]), dtype=torch.float32)
    truth = evaluation.case_type.isin(["rare", "critical", "adversarial"]).to_numpy()
    novel = evaluation.is_ood.to_numpy()
    torch.manual_seed(seed)
    started = time.perf_counter()
    with GpuSemaphore():
        model = Autoencoder(len(FEATURES)).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        model.train()
        device_train = train_x.cuda()
        for _ in range(150):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(device_train), device_train)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_score = ((model(device_train) - device_train) ** 2).mean(dim=1).cpu().numpy()
            device_eval = evaluation_x.cuda()
            score = ((model(device_eval) - device_eval) ** 2).mean(dim=1).cpu().numpy()
        device = str(next(model.parameters()).device)
        del model, optimizer, device_train, device_eval
        torch.cuda.empty_cache()
    threshold = float(np.quantile(train_score, 0.95))
    records = []
    predicted = score >= threshold
    for split_name, mask in {"validation": ~evaluation.is_ood.to_numpy(), "ood": evaluation.is_ood.to_numpy()}.items():
        records.append(
            {
                "experiment_id": f"anomaly-U3_pytorch_autoencoder-{split_name}-s{seed}",
                "config_hash": hashlib.sha256(
                    canonical_json({"model": "U3_pytorch_autoencoder", "seed": seed, "epochs": 150})
                ).hexdigest(),
                "dataset_hash": sha256_file(cases_path),
                "split_identifier": split_name,
                "seed": seed,
                "hardware_runtime": f"{platform.system()}-{platform.machine()}-cuda:{torch.cuda.get_device_name(0)}",
                "timestamp": utc_now(),
                "status": "ok",
                "runtime_seconds": time.perf_counter() - started,
                "metrics": json.dumps(_metrics(truth[mask], score[mask], predicted[mask], novel[mask]), sort_keys=True),
                "device": device,
            }
        )
    return records


def run() -> str:
    paths = ProjectPaths.discover()
    source = paths.root / "data" / "silver" / "synthetic_cases_development.parquet"
    with PhaseRun("P13", paths) as phase:
        anomaly = run_anomaly_validation(source)
        anomaly = pd.concat([anomaly, pd.DataFrame(run_autoencoder_validation(source))], ignore_index=True)
        anomaly_target = paths.root / "results" / "anomaly.parquet"
        anomaly.to_parquet(anomaly_target, index=False)
        from controlflow.modeling.semi_supervised import run_label_efficiency

        semi = pd.concat([run_label_efficiency(source, seed) for seed in (17, 29, 43)], ignore_index=True)
        semi.to_parquet(paths.root / "results" / "semi_supervised.parquet", index=False)
        phase.register(anomaly_target, "result_table")
        phase.register(paths.root / "results" / "semi_supervised.parquet", "result_table")
    return str(anomaly_target)


if __name__ == "__main__":
    print(run())
