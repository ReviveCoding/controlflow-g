from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Annotated, Any, cast

import httpx
import typer
import yaml
from filelock import FileLock
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from controlflow.core.state import ProjectPaths, atomic_write_json, sha256_file, utc_now

app = typer.Typer(no_args_is_help=True)


class RetryableHttpError(RuntimeError):
    pass


def _source_config(paths: ProjectPaths) -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load((paths.root / "configs/data/sources.yaml").read_text(encoding="utf-8")))


def _user_agent(config: dict[str, Any]) -> str:
    configured = str(config["user_agent"])
    contact = os.environ.get("CONTROLFLOW_CONTACT_EMAIL", "").strip()
    if contact:
        return f"ControlFlow-G public-data research {contact}"
    return configured


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError, RetryableHttpError)),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)
def download_atomic(url: str, target: Path, *, user_agent: str, timeout: float = 60.0) -> Path:
    """Cached, resumable HTTP download with an atomic final rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    partial = target.with_suffix(target.suffix + ".tmp")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": user_agent}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with httpx.stream("GET", url, headers=headers, timeout=timeout, follow_redirects=True) as response:
        if offset and response.status_code == 200:
            partial.unlink()
            offset = 0
        if response.status_code == 429 or response.status_code >= 500:
            raise RetryableHttpError(f"retryable HTTP {response.status_code} for {url}")
        response.raise_for_status()
        mode = "ab" if offset and response.status_code == 206 else "wb"
        with partial.open(mode) as handle:
            for chunk in response.iter_bytes(1024 * 1024):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(partial, target)
    return target


def record_dataset(
    *, name: str, official_source: str, snapshot: str, method: str, path: Path, terms: str
) -> dict[str, object]:
    paths = ProjectPaths.discover()
    manifest_path = paths.state / "data_manifest.json"
    relative = path.resolve().relative_to(paths.root).as_posix()
    current_hash = sha256_file(path)
    record: dict[str, object] = {
        "name": name,
        "official_source": official_source,
        "retrieval_timestamp": utc_now(),
        "snapshot": snapshot,
        "license_terms_reference": terms,
        "schema": None,
        "file_count": 1,
        "row_count": None,
        "size_bytes": path.stat().st_size,
        "sha256": current_hash,
        "download_method": method,
        "local_path": relative,
    }
    with FileLock(str(manifest_path) + ".lock"):
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior = next((item for item in manifest["datasets"] if item["name"] == name), None)
        if prior is not None and prior.get("local_path") == relative and prior.get("sha256") != current_hash:
            raise RuntimeError(f"immutable cached dataset hash mismatch for {name}; refusing to bless changed bytes")
        existing = [item for item in manifest["datasets"] if item["name"] != name]
        existing.append(record)
        manifest.update(updated_at=utc_now(), datasets=sorted(existing, key=lambda item: item["name"]))
        atomic_write_json(manifest_path, manifest)
    return record


@app.command("source")
def source(name: Annotated[str, typer.Argument(help="Configured source name")]) -> None:
    paths = ProjectPaths.discover()
    config = _source_config(paths)
    sources = config["sources"]
    if name not in sources:
        raise typer.BadParameter(f"Unknown source {name!r}")
    item = sources[name]
    if "url" not in item:
        raise typer.BadParameter(f"Source {name!r} requires a dated acquisition command")
    url = str(item["url"])
    suffix = Path(httpx.URL(url).path).suffix or ".bin"
    target = paths.root / "data" / "raw" / name / f"source{suffix}"
    download_atomic(
        url,
        target,
        user_agent=_user_agent(config),
        timeout=float(config["timeout_seconds"]),
    )
    record_dataset(
        name=name,
        official_source=url,
        snapshot=str(item["snapshot"]),
        method="HTTP GET with retry, resume, cache, and atomic rename",
        path=target,
        terms="See reports/00_source_manifest.md",
    )
    typer.echo(f"{target} sha256={sha256_file(target)}")


@app.command("ecfr")
def ecfr(dates: Annotated[list[str], typer.Option("--date")]) -> None:
    paths = ProjectPaths.discover()
    config = _source_config(paths)
    item = config["sources"]["ecfr_title_12"]
    for date in dates:
        url = str(item["url_template"]).format(date=date)
        target = paths.root / "data" / "raw" / "ecfr_title_12" / f"title-12-{date}.xml"
        download_atomic(url, target, user_agent=_user_agent(config), timeout=60)
        record_dataset(
            name=f"ecfr_title_12_{date}",
            official_source=url,
            snapshot=date,
            method="eCFR Versioner v1 full-title XML",
            path=target,
            terms="US government work; see reports/00_source_manifest.md",
        )
        time.sleep(0.5)


@app.command("verify")
def verify() -> None:
    paths = ProjectPaths.discover()
    manifest = json.loads((paths.state / "data_manifest.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for record in manifest["datasets"]:
        path = paths.root / record["local_path"]
        actual = sha256_file(path) if path.exists() else "MISSING"
        if actual != record["sha256"]:
            failures.append(f"{record['name']}: expected {record['sha256']}, got {actual}")
    if failures:
        typer.echo("\n".join(failures), err=True)
        raise typer.Exit(code=1)
    typer.echo(f"verified={len(manifest['datasets'])}")


@app.command("sec-filings")
def sec_filings(
    tickers: Annotated[list[str] | None, typer.Option("--ticker")] = None,
    max_filings: Annotated[int, typer.Option("--max-filings", min=1, max=100)] = 12,
) -> None:
    paths = ProjectPaths.discover()
    config = _source_config(paths)
    agent = _user_agent(config)
    if "example.invalid" in agent:
        raise typer.BadParameter("Set CONTROLFLOW_CONTACT_EMAIL to a real administrative contact for SEC access")
    ticker_path = paths.root / "data" / "raw" / "sec_company_tickers" / "source.json"
    mapping = json.loads(ticker_path.read_text(encoding="utf-8"))
    field_positions = {name: index for index, name in enumerate(mapping["fields"])}
    requested = {ticker.upper() for ticker in (tickers or ["JPM", "BAC", "WFC"])}
    companies = [row for row in mapping["data"] if row[field_positions["ticker"]].upper() in requested]
    candidates: list[dict[str, str | int]] = []
    for company in companies:
        cik = int(company[field_positions["cik"]])
        ticker = str(company[field_positions["ticker"]]).upper()
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        submissions_target = paths.root / "data" / "raw" / "sec_filings" / ticker / "submissions.json"
        download_atomic(submissions_url, submissions_target, user_agent=agent, timeout=60)
        record_dataset(
            name=f"sec_submissions_{ticker}",
            official_source=submissions_url,
            snapshot="retrieval-date",
            method="SEC Submissions JSON; declared User-Agent; 2 requests/second",
            path=submissions_target,
            terms="SEC public EDGAR access; see reports/00_source_manifest.md",
        )
        recent = json.loads(submissions_target.read_text(encoding="utf-8"))["filings"]["recent"]
        for index, form in enumerate(recent["form"]):
            if form not in {"10-K", "10-Q"}:
                continue
            candidates.append(
                {
                    "ticker": ticker,
                    "cik": cik,
                    "form": form,
                    "filing_date": recent["filingDate"][index],
                    "accession": recent["accessionNumber"][index],
                    "primary_document": recent["primaryDocument"][index],
                }
            )
        time.sleep(0.5)
    candidates.sort(key=lambda item: (str(item["filing_date"]), str(item["ticker"])), reverse=True)
    selected = candidates[:max_filings]
    for item in selected:
        accession_path = str(item["accession"]).replace("-", "")
        primary = str(item["primary_document"])
        url = f"https://www.sec.gov/Archives/edgar/data/{item['cik']}/{accession_path}/{primary}"
        safe_primary = re.sub(r"[^A-Za-z0-9._-]", "_", primary)
        target = (
            paths.root
            / "data"
            / "raw"
            / "sec_filings"
            / str(item["ticker"])
            / f"{item['filing_date']}_{item['form']}_{safe_primary}"
        )
        download_atomic(url, target, user_agent=agent, timeout=60)
        record_dataset(
            name=f"sec_{item['ticker']}_{item['form']}_{item['filing_date']}_{item['accession']}",
            official_source=url,
            snapshot=str(item["filing_date"]),
            method="Bounded SEC EDGAR primary-document acquisition; declared User-Agent; 2 requests/second",
            path=target,
            terms="Public EDGAR filing; see reports/00_source_manifest.md",
        )
        typer.echo(f"{target} sha256={sha256_file(target)}")
        time.sleep(0.5)


@app.command("govinfo")
def govinfo(
    year: Annotated[int, typer.Option("--year", min=1996, max=2100)] = 2024,
    volumes: Annotated[int, typer.Option("--volumes", min=1, max=20)] = 10,
) -> None:
    paths = ProjectPaths.discover()
    config = _source_config(paths)
    item = config["sources"]["govinfo_title_12"]
    for volume in range(1, volumes + 1):
        url = str(item["url_template"]).format(year=year, volume=volume)
        target = paths.root / "data" / "raw" / "govinfo_title_12" / str(year) / f"CFR-{year}-title12-vol{volume}.xml"
        download_atomic(url, target, user_agent=_user_agent(config), timeout=60)
        record_dataset(
            name=f"govinfo_title_12_{year}_vol{volume}",
            official_source=url,
            snapshot=f"annual-{year}-01-01",
            method="Official GovInfo CFR bulk XML",
            path=target,
            terms="US government work; embedded third-party caveat; see reports/00_source_manifest.md",
        )
        typer.echo(f"{target} sha256={sha256_file(target)}")


if __name__ == "__main__":
    app()
