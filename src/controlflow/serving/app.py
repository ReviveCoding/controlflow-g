from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="ControlFlow-G", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "mode": "production-like-simulation", "external_actions": "disabled"}
