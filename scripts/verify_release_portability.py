"""Verify an additive EOL attestation against the immutable V2.6 release tag.

The original freeze manifest and receipts retain their exact byte checks. This
script proves that the tagged Git blobs reconstruct those original bytes, then
runs the original receipt verifier in a disposable tree.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from controlflow.v26.closure_receipt import verify_receipt

ROOT = Path(__file__).resolve().parents[1]
ATTESTATION = ROOT / "reports/ci_portability/v26_eol_attestation.json"


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _digest(data: bytes) -> tuple[str, int]:
    return hashlib.sha256(data).hexdigest(), len(data)


def _restore(blob: bytes, item: dict[str, Any]) -> bytes:
    if b"\r" in blob or blob.count(b"\n") != item["newline_count"]:
        raise ValueError(f"Git blob is not the attested LF representation: {item['path']}")
    if item["mode"] == "all_crlf":
        return blob.replace(b"\n", b"\r\n")
    if item["mode"] != "selected_crlf":
        raise ValueError(f"unknown EOL transform: {item['path']}")
    selected: set[int] = set()
    previous = -1
    for start, end in item["crlf_line_ranges"]:
        if not (isinstance(start, int) and isinstance(end, int) and previous < start <= end < item["newline_count"]):
            raise ValueError(f"invalid EOL range: {item['path']}")
        selected.update(range(start, end + 1))
        previous = end
    parts = blob.split(b"\n")
    return (
        b"".join(part + (b"\r\n" if index in selected else b"\n") for index, part in enumerate(parts[:-1])) + parts[-1]
    )


def main() -> None:
    attestation = json.loads(ATTESTATION.read_text(encoding="utf-8"))
    if attestation["schema_version"] != 1:
        raise ValueError("unsupported attestation schema")
    ref = attestation["source_ref"]
    if ref != "controlflow-g-v26-promote":
        raise ValueError("unexpected release ref")
    if _git("rev-parse", f"refs/tags/{ref}").decode().strip() != attestation["tag_object_sha"]:
        raise ValueError("release tag object changed")
    if _git("rev-parse", f"{ref}^{{commit}}").decode().strip() != attestation["release_commit_sha"]:
        raise ValueError("release tag target changed")

    manifest_path = attestation["freeze_manifest_path"]
    manifest_blob = _git("show", f"{ref}:{manifest_path}")
    if _digest(manifest_blob)[0] != attestation["freeze_manifest_sha256"]:
        raise ValueError("tagged freeze manifest changed")
    if (ROOT / manifest_path).read_bytes() != manifest_blob:
        raise ValueError("current freeze manifest differs from promoted release")
    manifest = json.loads(manifest_blob)
    rows = manifest["bindings"]
    if len(rows) != attestation["bound_file_count"]:
        raise ValueError("freeze binding count changed")

    transforms = {item["path"]: item for item in attestation["transforms"]}
    if len(transforms) != len(attestation["transforms"]):
        raise ValueError("duplicate attestation path")
    used: set[str] = set()
    direct = 0
    for row in rows:
        path = row["path"]
        blob = _git("show", f"{ref}:{path}")
        expected = row["sha256"], row["size"]
        if path not in transforms:
            if _digest(blob) != expected:
                raise ValueError(f"unattested frozen byte mismatch: {path}")
            direct += 1
            continue
        item = transforms[path]
        used.add(path)
        if _digest(blob) != (item["git_blob_sha256"], item["git_blob_size"]):
            raise ValueError(f"tagged Git blob mismatch: {path}")
        if expected != (item["frozen_sha256"], item["frozen_size"]):
            raise ValueError(f"frozen binding mismatch: {path}")
        if _digest(blob) == expected or _digest(_restore(blob, item)) != expected:
            raise ValueError(f"invalid EOL attestation: {path}")
    receipt_results: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix=".v26_portability_", dir=ROOT) as temporary:
        for stage in ("v26_development_rehearsal_26015", "v26_qualification", "v26_final"):
            graph_name = f"state/{stage}_bindings.json"
            receipt_name = f"state/{stage}_closure_receipt.json"
            graph = json.loads(_git("show", f"{ref}:{graph_name}"))
            receipt = json.loads(_git("show", f"{ref}:{receipt_name}"))
            binding_rows = [
                graph["rules"],
                *graph["bindings"].values(),
                receipt["qualification_graph"],
                *receipt["deferred_bindings"],
            ]
            for row in binding_rows:
                name = row["path"]
                if name in transforms:
                    used.add(name)
                    if (row["sha256"], row["size"]) != (
                        transforms[name]["frozen_sha256"],
                        transforms[name]["frozen_size"],
                    ):
                        raise ValueError(f"receipt and EOL attestation disagree: {name}")
            required = {
                graph_name,
                receipt_name,
                graph["rules"]["path"],
                *(row["path"] for row in graph["bindings"].values()),
                *(row["path"] for row in receipt["deferred_bindings"]),
            }
            root = Path(temporary) / stage
            for name in sorted(required):
                blob = _git("show", f"{ref}:{name}")
                content = _restore(blob, transforms[name]) if name in transforms else blob
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            result = verify_receipt(root, root / graph_name, root / receipt_name)
            receipt_results[stage] = result["status"]
            if result["status"] != "PASS":
                raise ValueError(f"reconstructed receipt failed: {stage}: {result['failures']}")
    if used != set(transforms):
        raise ValueError("attestation contains unverified paths")

    print(
        json.dumps(
            {
                "status": "PASS",
                "source_ref": ref,
                "release_commit": attestation["release_commit_sha"],
                "bound_files": len(rows),
                "git_byte_matches": direct,
                "eol_reconstructions": len(transforms),
                "reconstructed_receipts": receipt_results,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
