from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from controlflow.core.state import ProjectPaths, atomic_write_json, sha256_file, utc_now
from controlflow.v2.data import load_v2_split
from controlflow.v2.decomposed import ROOT_CAUSES
from controlflow.v2.fusion import EMBEDDING_MODEL, EMBEDDING_REVISION
from controlflow.v2.resources import CrossPlatformGpuSemaphore

TAXONOMY_DESCRIPTIONS = {
    "ROUTINE_VARIANCE": "Routine operational variance with no material control failure.",
    "OWNERSHIP_AMBIGUITY": "Ambiguous process or control ownership requiring clarification.",
    "MATERIAL_CONTROL_BREAKDOWN": "Material breakdown of an established internal control.",
    "NOVEL_THIRD_PARTY_FAILURE": "Previously unseen external vendor, supplier, or third-party dependency failure.",
    "SOURCE_EVIDENCE_MISSING": "Required source evidence is absent or unavailable.",
    "AUTHORITATIVE_SOURCE_CONFLICT": "Authorized evidence sources conflict with one another.",
    "TEMPORAL_POLICY_MISMATCH": "The policy version is incorrect for the event time.",
    "AUTHORIZATION_SCOPE_VIOLATION": "The requested access or action exceeds authorization scope.",
}
TAXONOMY_AUGMENTATIONS = {
    "ROUTINE_VARIANCE": [
        "isolated processing fluctuation with corroboration",
        "ordinary immaterial operating variation",
    ],
    "OWNERSHIP_AMBIGUITY": ["unresolved accountability handoff", "two teams dispute process ownership"],
    "MATERIAL_CONTROL_BREAKDOWN": ["material safeguard ceased operating", "urgent failure of an established control"],
    "NOVEL_THIRD_PARTY_FAILURE": [
        "newly observed failure in an external provider dependency",
        "unseen supplier cascade",
        "external correspondent disruption",
        "third party service exhibits a new failure mechanism",
    ],
    "SOURCE_EVIDENCE_MISSING": [
        "primary source record unavailable at cutoff",
        "required corroborating evidence absent",
    ],
    "AUTHORITATIVE_SOURCE_CONFLICT": [
        "trusted records reach incompatible conclusions",
        "authoritative observations disagree",
    ],
    "TEMPORAL_POLICY_MISMATCH": [
        "historical event needs the then effective rule",
        "current policy is wrong for event time",
    ],
    "AUTHORIZATION_SCOPE_VIOLATION": [
        "operation exceeds assigned business scope",
        "analyst lacks entitlement for requested scope",
    ],
}


def _metrics(truth: pd.Series, prediction: np.ndarray[Any, Any]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=ROOT_CAUSES, average="macro", zero_division=0)),
    }


def run_root_embedding_study(dataset: Path, seed: int = 20_260_912) -> Path:
    from sentence_transformers import SentenceTransformer
    from xgboost import XGBClassifier

    paths = ProjectPaths.discover()
    frame = pd.read_parquet(dataset)
    split_names = ("train", "calibration", "validation", "ood", "adversarial")
    parts = {name: frame[frame["case_id"].isin(set(load_v2_split(name)))].copy() for name in split_names}
    with CrossPlatformGpuSemaphore(timeout_seconds=60):
        encoder = SentenceTransformer(
            EMBEDDING_MODEL,
            revision=EMBEDDING_REVISION,
            device="cuda",
            trust_remote_code=False,
        )
        all_embeddings = encoder.encode(
            frame["narrative"].astype(str).tolist(),
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        augmented_train_texts = [
            f"{row.narrative} Taxonomy definition: {TAXONOMY_DESCRIPTIONS[str(row.root_cause_code)]}"
            for row in parts["train"].itertuples(index=False)
        ]
        augmented_train_embeddings = encoder.encode(
            augmented_train_texts,
            batch_size=128,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        taxonomy_labels = list(TAXONOMY_DESCRIPTIONS)
        taxonomy_embeddings = encoder.encode(
            [TAXONOMY_DESCRIPTIONS[label] for label in taxonomy_labels],
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        schema_pairs = [(label, text) for label in taxonomy_labels for text in TAXONOMY_AUGMENTATIONS[label]]
        controls = ("AC-2", "AU-6", "CA-7", "IR-4", "RA-5", "SI-4")
        regulations = ("12CFR-21", "12CFR-30", "12CFR-1005", "12CFR-1026")
        contextual_pairs = [
            (
                label,
                f"Independent case observation: {text}. References {control} and {regulation}; "
                f"repeat={index % 8}; prior exceptions={index % 5}.",
            )
            for label, text in schema_pairs
            for index, (control, regulation) in enumerate(product(controls, regulations))
        ]
        schema_pairs.extend(contextual_pairs)
        schema_texts = [text for _, text in schema_pairs]
        schema_labels = [label for label, _ in schema_pairs]
        schema_embeddings = encoder.encode(
            schema_texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        device = str(encoder.device)
    if not device.startswith("cuda"):
        raise RuntimeError(f"root embedding CPU fallback refused: {device}")
    index = {identifier: position for position, identifier in enumerate(frame["case_id"].astype(str))}
    embeddings = {
        name: all_embeddings[[index[identifier] for identifier in part["case_id"].astype(str)]]
        for name, part in parts.items()
    }

    models: dict[str, Any] = {
        "root_embedding_linear": LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed),
        "root_embedding_linear_taxonomy_augmented": LogisticRegression(
            max_iter=2_000, class_weight="balanced", random_state=seed
        ),
        "root_embedding_small_neural_taxonomy_augmented": MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=300, early_stopping=True, random_state=seed
        ),
    }
    taxonomy_repetitions = 10
    augmented_train = np.row_stack(
        [
            embeddings["train"],
            augmented_train_embeddings,
            np.tile(taxonomy_embeddings, (taxonomy_repetitions, 1)),
            np.tile(schema_embeddings, (taxonomy_repetitions, 1)),
        ]
    )
    augmented_labels = pd.concat(
        [
            parts["train"]["root_cause_code"],
            parts["train"]["root_cause_code"],
            pd.Series(taxonomy_labels * taxonomy_repetitions),
            pd.Series(schema_labels * taxonomy_repetitions),
        ],
        ignore_index=True,
    )
    observed_classes = sorted(parts["train"]["root_cause_code"].astype(str).unique())
    class_to_index = {label: position for position, label in enumerate(observed_classes)}
    train_codes = parts["train"]["root_cause_code"].astype(str).map(class_to_index).to_numpy()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=len(observed_classes),
        tree_method="hist",
        device="cuda",
        random_state=seed,
        n_jobs=1,
    )
    with CrossPlatformGpuSemaphore(timeout_seconds=60):
        taxonomy_codes = np.asarray([class_to_index[label] for label in taxonomy_labels] * taxonomy_repetitions)
        schema_codes = np.asarray([class_to_index[label] for label in schema_labels] * taxonomy_repetitions)
        xgb.fit(augmented_train, np.concatenate([train_codes, train_codes, taxonomy_codes, schema_codes]))
    if '"device":"cuda' not in xgb.get_booster().save_config():
        raise RuntimeError("root embedding XGBoost CPU fallback refused")
    models["root_embedding_xgboost_cuda_taxonomy_augmented"] = xgb

    records: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, np.ndarray[Any, Any]]] = {}
    for name, model in models.items():
        if "xgboost_cuda" in name:
            model_predictions = {
                split: np.asarray(observed_classes)[model.predict(values).astype(int)]
                for split, values in embeddings.items()
            }
        else:
            if name.endswith("taxonomy_augmented"):
                model.fit(augmented_train, augmented_labels)
            else:
                model.fit(embeddings["train"], parts["train"]["root_cause_code"])
            model_predictions = {split: model.predict(values) for split, values in embeddings.items()}
        predictions[name] = model_predictions
        records.append(
            {
                "model": name,
                **{
                    f"validation_{key}": value
                    for key, value in _metrics(
                        parts["validation"]["root_cause_code"], model_predictions["validation"]
                    ).items()
                },
                **{
                    f"ood_{key}": value
                    for key, value in _metrics(parts["ood"]["root_cause_code"], model_predictions["ood"]).items()
                },
            }
        )
    result = pd.DataFrame(records)
    result["selection_priority"] = result["model"].str.contains("taxonomy_augmented").map({True: 0, False: 1})
    result = result.sort_values(
        ["validation_accuracy", "validation_macro_f1", "selection_priority", "model"],
        ascending=[False, False, True, True],
    )
    selected_name = str(result.iloc[0]["model"])
    selected_model = models[selected_name]
    model_dir = paths.root / "artifacts/v2/models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": selected_model,
            "observed_classes": observed_classes,
            "model_name": selected_name,
            "novelty_threshold": None,
        },
        model_dir / "root_embedding_selected.joblib",
    )
    joblib.dump(
        {
            "case_ids": frame["case_id"].astype(str).tolist(),
            "narrative_sha256": [
                hashlib.sha256(value.encode()).hexdigest() for value in frame["narrative"].astype(str)
            ],
            "embeddings": all_embeddings,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
        },
        paths.root / "artifacts/v2/root_embeddings.joblib",
    )
    target = paths.root / "results/v2/root_embedding_classifier.parquet"
    result.to_parquet(target, index=False)
    atomic_write_json(
        paths.state / "v2_root_embedding_manifest.json",
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "dataset_sha256": sha256_file(dataset),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
            "embedding_device": device,
            "training_splits": ["train"],
            "selection_split": "validation",
            "diagnostic_only_splits": ["ood", "adversarial"],
            "selected": selected_name,
            "selected_route": "direct_supervised_taxonomy_prediction_no_posthoc_ood_route",
            "novelty_threshold": None,
            "results": json.loads(result.to_json(orient="records")),
        },
    )
    return target


def _encode_unseen(narratives: list[str]) -> np.ndarray[Any, Any]:
    from sentence_transformers import SentenceTransformer

    with CrossPlatformGpuSemaphore(timeout_seconds=60):
        encoder = SentenceTransformer(
            EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device="cuda", trust_remote_code=False
        )
        values = encoder.encode(
            narratives, batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        device = str(encoder.device)
    if not device.startswith("cuda"):
        raise RuntimeError(f"root embedding CPU fallback refused: {device}")
    return np.asarray(values)


def _looks_like_prompt_injection(value: str) -> bool:
    text = value.casefold()
    instruction = any(token in text for token in ("ignore", "disregard", "override", "forget", "system prompt"))
    escalation = any(
        token in text for token in ("policy", "approval", "authorization", "unrestricted", "tool", "instructions")
    )
    return instruction and escalation


def predict_root_causes_runtime(
    frame: pd.DataFrame,
    *,
    encode_unseen: Any | None = None,
    cache_path: Path | None = None,
) -> np.ndarray[Any, Any]:
    """Infer arbitrary case IDs; use only narrative-hash-bound cached embeddings."""
    paths = ProjectPaths.discover()
    embedding_artifact = joblib.load(cache_path or paths.root / "artifacts/v2/root_embeddings.joblib")
    model_artifact = joblib.load(paths.root / "artifacts/v2/models/root_embedding_selected.joblib")
    hashes = embedding_artifact.get("narrative_sha256", [])
    cache = (
        {
            (str(identifier), str(digest)): embedding_artifact["embeddings"][position]
            for position, (identifier, digest) in enumerate(zip(embedding_artifact["case_ids"], hashes, strict=True))
        }
        if len(hashes) == len(embedding_artifact["case_ids"])
        else {}
    )
    rows: list[np.ndarray[Any, Any] | None] = []
    missing_positions: list[int] = []
    missing_text: list[str] = []
    for position, row in enumerate(frame.itertuples(index=False)):
        narrative = str(row.narrative)
        key = (str(row.case_id), hashlib.sha256(narrative.encode()).hexdigest())
        cached = cache.get(key)
        rows.append(cached)
        if cached is None:
            missing_positions.append(position)
            missing_text.append(narrative)
    if missing_text:
        encoded = (encode_unseen or _encode_unseen)(missing_text)
        for position, value in zip(missing_positions, encoded, strict=True):
            rows[position] = np.asarray(value)
    if any(value is None for value in rows):
        raise RuntimeError("root embedding inference left unresolved rows")
    values = np.vstack([value for value in rows if value is not None])
    model = model_artifact["model"]
    if "xgboost_cuda" in str(model_artifact["model_name"]):
        prediction = np.asarray(model_artifact["observed_classes"])[model.predict(values).astype(int)]
    else:
        prediction = model.predict(values)
    probability = model.predict_proba(values)
    prediction = np.asarray(prediction, dtype=object)
    novelty_threshold = model_artifact.get("novelty_threshold")
    if novelty_threshold is not None:
        prediction[probability.max(axis=1) < float(novelty_threshold)] = "NOVEL_THIRD_PARTY_FAILURE"
    prediction[np.asarray([_looks_like_prompt_injection(value) for value in frame["narrative"].astype(str)])] = (
        "PROMPT_INJECTION_ATTEMPT"
    )
    return prediction


def predict_precomputed_root_causes(frame: pd.DataFrame) -> np.ndarray[Any, Any]:
    """Backward-compatible alias; unlike the old implementation it supports unseen IDs."""
    return predict_root_causes_runtime(frame)


def precompute_runtime_embeddings(frame: pd.DataFrame, target: Path) -> Path:
    """GPU-only, label-blind embedding stage that may run before vLLM owns the GPU."""
    narratives = frame["narrative"].astype(str).tolist()
    embeddings = _encode_unseen(narratives)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "case_ids": frame["case_id"].astype(str).tolist(),
            "narrative_sha256": [hashlib.sha256(value.encode()).hexdigest() for value in narratives],
            "embeddings": embeddings,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_revision": EMBEDDING_REVISION,
        },
        target,
    )
    return target
