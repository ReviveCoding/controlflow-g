# Artifact closure

The preliminary binding graph contains 33 named file bindings (`state/v24_qualification_bindings.json`); a read-only binding check found no missing or changed bytes, and the unbound scan was clean (`state/v24_unbound_artifacts.json`). The post-close verifier failed before gate calculation and terminal decision receipt. Thus the complete decision artifact closure was not achieved, regardless of the preliminary bindings (`state/v24_postclose_verification.json`).
