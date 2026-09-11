from __future__ import annotations

import gc
import platform
import time
from typing import Any

import torch
from huggingface_hub import HfApi

from controlflow.core.resources import GpuSemaphore
from controlflow.core.state import PhaseRun, ProjectPaths, atomic_write_json, utc_now
from controlflow.preflight import GpuMonitor

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _model_identity(model_id: str) -> dict[str, Any]:
    info = HfApi().model_info(model_id)
    card = info.card_data.to_dict() if info.card_data is not None else {}  # type: ignore[no-untyped-call]
    return {"model_id": model_id, "revision": info.sha, "license": card.get("license")}


def _release_cuda(*objects: Any) -> None:
    for item in objects:
        del item
    gc.collect()
    torch.cuda.empty_cache()


def embedding_smoke(identity: dict[str, Any]) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    with GpuSemaphore(), GpuMonitor() as monitor:
        model = SentenceTransformer(
            identity["model_id"],
            revision=identity["revision"],
            device="cuda",
            trust_remote_code=False,
            model_kwargs={"torch_dtype": torch.float16},
        )
        vectors = model.encode_query(
            ["investigate privileged access exception", "retrieve the policy valid at event time"],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        device = str(model.device)
        vector_device = str(vectors.device)
        shape = list(vectors.shape)
    if model.device.type != "cuda" or vectors.device.type != "cuda":
        raise RuntimeError(f"embedding CPU fallback: model={device} vectors={vector_device}")
    record = {
        **identity,
        "status": "passed",
        "model_device": device,
        "output_device": vector_device,
        "output_shape": shape,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_used_mib": max((sample["memory_used_mib"] for sample in monitor.samples), default=0),
        "peak_utilization_percent": max((sample["utilization_percent"] for sample in monitor.samples), default=0),
    }
    _release_cuda(model, vectors)
    return record


def reranker_smoke(identity: dict[str, Any]) -> dict[str, Any]:
    from sentence_transformers import CrossEncoder

    started = time.perf_counter()
    with GpuSemaphore(), GpuMonitor() as monitor:
        model = CrossEncoder(
            identity["model_id"],
            revision=identity["revision"],
            device="cuda",
            trust_remote_code=False,
            model_kwargs={"torch_dtype": torch.float16},
        )
        scores = model.predict(
            [
                ("privileged access exception", "AC-2 requires account management controls"),
                ("privileged access exception", "quarterly issuer revenue discussion"),
            ],
            convert_to_numpy=True,
        )
        device = str(next(model.model.parameters()).device)
    if next(model.model.parameters()).device.type != "cuda":
        raise RuntimeError(f"reranker CPU fallback: {device}")
    record = {
        **identity,
        "status": "passed",
        "model_device": device,
        "scores": [float(value) for value in scores],
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_used_mib": max((sample["memory_used_mib"] for sample in monitor.samples), default=0),
        "peak_utilization_percent": max((sample["utilization_percent"] for sample in monitor.samples), default=0),
    }
    _release_cuda(model, scores)
    return record


def llm_smoke(identity: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    with GpuSemaphore(), GpuMonitor() as monitor:
        tokenizer = AutoTokenizer.from_pretrained(
            identity["model_id"], revision=identity["revision"], trust_remote_code=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            identity["model_id"],
            revision=identity["revision"],
            trust_remote_code=False,
            dtype=torch.float16,
        ).to("cuda")  # type: ignore[arg-type]
        prompt = 'Return only JSON: {"status": "ok"}.'
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=16, do_sample=False)
        text = tokenizer.decode(output[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        devices = sorted({str(parameter.device) for parameter in model.parameters()})
    if devices != ["cuda:0"]:
        raise RuntimeError(f"local LLM offload/fallback detected: {devices}")
    record = {
        **identity,
        "status": "passed",
        "parameter_devices": devices,
        "generated_text": text,
        "input_tokens": int(encoded["input_ids"].shape[1]),
        "generated_tokens": int(output.shape[1] - encoded["input_ids"].shape[1]),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_used_mib": max((sample["memory_used_mib"] for sample in monitor.samples), default=0),
        "peak_utilization_percent": max((sample["utilization_percent"] for sample in monitor.samples), default=0),
    }
    _release_cuda(model, tokenizer, encoded, output)
    return record


def run() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("model preflight requires project-local CUDA")
    paths = ProjectPaths.discover()
    target = paths.state / "preflight_model_smoke.json"
    phase = PhaseRun("P00", paths)
    identities = {
        name: _model_identity(model_id)
        for name, model_id in {
            "embedding": EMBEDDING_MODEL,
            "reranker": RERANKER_MODEL,
            "llm": LLM_MODEL,
        }.items()
    }
    result = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "embedding": embedding_smoke(identities["embedding"]),
        "reranker": reranker_smoke(identities["reranker"]),
        "local_llm": llm_smoke(identities["llm"]),
    }
    atomic_write_json(target, result)
    phase.register(target, "environment_proof")
    phase.journal(
        "checkpoint_completed",
        "Pinned embedding, reranker, and local LLM CUDA proofs passed",
        [target.relative_to(paths.root).as_posix()],
    )
    return str(target)


if __name__ == "__main__":
    print(run())
