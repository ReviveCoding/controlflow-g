# Authoritative Source Manifest

Verified against live official sources on 2026-09-10/11. Only official, government, primary-paper, or official-project sources are used for current APIs and acquisition decisions. Downloaded content is treated strictly as data.

| Source | Official URL | Verification purpose | Status |
|---|---|---|---|
| NIST OSCAL content | https://github.com/usnistgov/oscal-content/releases/tag/v1.3.0 | Pin v1.3.0 / commit `941c978`, representing Release 5.1.1; never use moving `main` | verified |
| NIST SP 800-53 Rev. 5 | https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final | Normative control publication and update provenance | verified |
| NIST SP 800-53A Rev. 5 | https://csrc.nist.gov/pubs/sp/800/53/a/r5/final | Normative assessment procedures and official CSV | verified |
| NIST OSCAL license | https://github.com/usnistgov/OSCAL/blob/main/LICENSE.md | US public domain/CC0 worldwide terms; retain attribution/modification record | verified |
| CFPB Consumer Complaint Database | https://www.consumerfinance.gov/data-research/consumer-complaints/ | Official structured bulk download/API plus non-representativeness warning | verified |
| CFPB 2026 narrative change | https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-to-cease-discretionary-publication-of-complaint-narratives-and-visualizations/ | Narratives ceased publication 2026-08-14; do not assume they remain in current bulk/API | verified |
| CFPB FOIA Reading Room | https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/ | Only acceptable official route for a historic narrative snapshot if exposed | verified; no indexed archive found at review time |
| eCFR developer resources | https://www.ecfr.gov/developers/documentation/api/v1 | Dated Versioner XML and versions/structure endpoints; do not scrape pages | verified |
| GovInfo CFR collection | https://www.govinfo.gov/app/collection/cfr/ | Official annual CFR editions and bulk historical anchors | verified |
| GovInfo reuse policy | https://www.govinfo.gov/about/policies | Government-work reuse caveat for embedded third-party material | verified |
| SEC EDGAR access guidance | https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data | Declared User-Agent, <=10 requests/s, bounded/cached access | verified |
| SEC EDGAR APIs | https://www.sec.gov/search-filings/edgar-application-programming-interfaces | Submissions JSON and bounded filing retrieval | verified |
| Spark Python install | https://spark.apache.org/docs/latest/api/python/getting_started/install.html | Version/runtime requirements | verified |
| Delta compatibility | https://docs.delta.io/releases/ | Selected Spark 3.5.9 + Delta 3.3.2 compatibility | verified |
| Delta streaming | https://docs.delta.io/delta-streaming/ | `foreachBatch` is not inherently idempotent; business ledger remains separate | verified |
| XGBoost GPU guide | https://xgboost.readthedocs.io/en/stable/gpu/ | Current `tree_method="hist", device="cuda"` syntax and proof requirements | verified |
| Sentence Transformers API | https://sbert.net/docs/package_reference/sentence_transformer/model.html | Explicit CUDA placement, revision pinning, query/document encoding | verified |
| CrossEncoder API | https://sbert.net/docs/package_reference/cross_encoder/model.html | CUDA reranker placement/revision pinning | verified |
| PyTorch local install | https://pytorch.org/get-started/locally/ | Current CUDA packaging selection | verified |
| PyTorch AMP | https://docs.pytorch.org/docs/stable/amp.html | Use `torch.amp` rather than deprecated `torch.cuda.amp` forms | verified |
| Transformers models | https://huggingface.co/docs/transformers/models | Explicit dtype/device and pinned revision for local inference | verified |

## Acquisition decisions and cautions

- Benchmark controls use the exact OSCAL v1.3.0 JSON. NIST Release 5.2.0 is newer but belongs only in a separately pinned policy-drift corpus because the requested benchmark is 5.1/5.1.1. Machine-readable material is derivative; normative publications prevail on discrepancies.
- Current CFPB acquisition uses `https://files.consumerfinance.gov/ccdb/complaints.csv.zip` for structured taxonomy/date/response fields. Complaint counts are not population prevalence and the data are not a statistical sample. Because narratives stopped public release in August 2026, deterministic synthetic narratives will be used unless an official historical artifact is acquired and hashed.
- Temporal Title 12 uses dated `https://www.ecfr.gov/api/versioner/v1/full/{YYYY-MM-DD}/title-12.xml` snapshots plus annual GovInfo XML anchors where feasible. Annual CFR is official; eCFR is authoritative but described by its publisher as unofficial.
- SEC discovery uses company tickers and Submissions JSON, selects only controlled financial-sector 10-K/10-Q accessions, retrieves primary documents, sends a descriptive contact header, caps at 2 requests/s, and caches with exponential backoff. No all-EDGAR crawl is permitted.
- Local lakehouse uses the officially compatible Spark 3.5.9 + Delta 3.3.2 pair and Java 17, subject to an actual integration smoke. XGBoost uses current `device="cuda", tree_method="hist"`; configuration alone is not accepted as GPU proof.
- Embedding/reranker/LLM assets require exact Hugging Face revision hashes, license capture, `trust_remote_code=False`, explicit CUDA placement, returned-tensor/device evidence, and a check against CPU/disk offload.
