# Reproducing the V2.6 research release checks

## Release boundaries

The frozen research release is the annotated tag `controlflow-g-v26-promote`, peeled commit `f21a4192373c5bcb815e12a7fa10ac8f183b482b`. The separate `controlflow-g-v26-public` tag identifies the later MIT-licensed distribution package. Verify both tags without changing them:

```sh
git rev-parse 'controlflow-g-v26-promote^{commit}'
git show --no-patch --format=fuller controlflow-g-v26-promote
git rev-parse 'controlflow-g-v26-public^{commit}'
```

The first command must print `f21a4192373c5bcb815e12a7fa10ac8f183b482b`. The public tag must peel to the later packaging commit; its [LICENSE](LICENSE) file is authoritative for distribution terms. Read [`state/v26_release_decision.json`](state/v26_release_decision.json), [`reports/v26/release.md`](reports/v26/release.md), [`results/v26/final/gate_decision.json`](results/v26/final/gate_decision.json), and [`results/v26/final/denominators.json`](results/v26/final/denominators.json) to inspect the frozen conclusion and metrics. **Do not rerun V26QUAL or V26FINAL as fresh evidence.** Their identities and holdouts are consumed.

## Environment and responsibilities

The package supports Python `>=3.11,<3.13`; portable GitHub Actions uses Python 3.11 ([`pyproject.toml`](pyproject.toml), [`.github/workflows/ci.yml`](.github/workflows/ci.yml)). The dependency graph is in `uv.lock`. Install `uv` and run from the repository root:

```sh
uv sync --locked --extra dev
```

Portable static checks and tests run on Windows or Linux without a local model server. The historical sealed execution used Windows for orchestration/artifact handling and WSL for local vLLM serving. The final environment manifest records Python 3.11.9, vLLM 0.29.0, `Qwen/Qwen3-4B-Instruct-2507` at revision `cdbee75f17c01a7cc42f958dc650907174af0554`, and a loopback server ([`state/v26_final_environment_manifest.json`](state/v26_final_environment_manifest.json), [`state/v26_freeze_manifest.json`](state/v26_freeze_manifest.json)). The historical local GPU was an NVIDIA RTX 4090 Laptop GPU. A new sealed/model-serving run would require a verified compatible NVIDIA GPU and the repository GPU semaphore, but it is outside this release-verification procedure.

## Portable validation

Run the same checks as CI from the repository root:

```sh
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run pytest -q -m "not gpu and not spark and not network and not sealed"
uv run --no-sync python scripts/verify_release_portability.py
```

The pytest marker expression excludes GPU, Spark, network, and sealed tests. These checks verify current code and release portability; they do not reproduce the historical one-shot final inference. For the exact CI definition, see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Frozen-byte portability

The promoted tag and original Windows CRLF-bound manifests and receipts are immutable. Git stores normalized LF blobs for some files. The additive [`reports/ci_portability/v26_eol_attestation.json`](reports/ci_portability/v26_eol_attestation.json) lists original size/hash, tagged blob size/hash, and the deterministic line-ending transform for each affected file. `scripts/verify_release_portability.py` checks the annotated tag object and peeled commit, validates each tagged blob, reconstructs the original bytes, and verifies the original receipt graphs in a disposable directory. It never rewrites the frozen evidence. Portability and documentation commits live after the promoted research commit on `v2.6-development`.

## Evidence integrity

The release decision records qualification and final manifest hashes, freeze hashes, gate and denominator hashes, closure receipt, review outcome, and terminal checks. The final freeze manifest and receipt bindings in `state/` define the exact byte-level evidence. Inspect them read-only. Regenerating a sealed cohort, changing gates, or rerunning a consumed identity would create a different study, not a validation of V2.6.
