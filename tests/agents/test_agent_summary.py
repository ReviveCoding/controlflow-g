from __future__ import annotations

import pandas as pd

from controlflow.agents.experiments import agent_hardware_runtime, summary_security_metrics


def test_unauthorized_action_rate_requires_an_executed_action() -> None:
    frame = pd.DataFrame(
        {
            "action_executed": [False, False, True, True],
            "authorization_correct": [False, True, False, True],
        }
    )
    metrics = summary_security_metrics(frame)

    assert metrics["authorization_decision_error_rate"] == 0.5
    assert metrics["unauthorized_action_rate"] == 0.25


def test_cpu_rules_runtime_is_not_labeled_cuda() -> None:
    names = ["AG0_rules_templates", "AG1_single_llm", "AG6_controlflow_g"]
    runtimes = {name: agent_hardware_runtime(name, "verified-device") for name in names}

    assert runtimes["AG0_rules_templates"] == "CPU:deterministic-rules"
    assert runtimes["AG1_single_llm"].startswith("CUDA:")
    assert runtimes["AG6_controlflow_g"].startswith("CUDA:")
