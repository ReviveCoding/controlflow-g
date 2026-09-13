from __future__ import annotations

from controlflow.v2.critical import run_critical_tournament
from controlflow.v2.data import create_development_dataset
from controlflow.v2.decomposed import run_decomposed_models
from controlflow.v2.fusion import run_fusion_study


def main() -> None:
    dataset = create_development_dataset()
    print(dataset)
    print(run_critical_tournament(dataset))
    print(run_fusion_study(dataset))
    print(run_decomposed_models(dataset))


if __name__ == "__main__":
    main()
