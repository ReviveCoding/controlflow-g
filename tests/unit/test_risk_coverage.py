import numpy as np
import pandas as pd

from controlflow.modeling.calibration import risk_coverage


def test_risk_coverage_preserves_critical_capture_and_residual_denominators():
    probabilities = np.array(
        [
            [0.03, 0.03, 0.04, 0.90],
            [0.70, 0.10, 0.10, 0.10],
            [0.60, 0.20, 0.10, 0.10],
            [0.80, 0.10, 0.05, 0.05],
        ]
    )
    result = risk_coverage(probabilities, np.array([3, 3, 0, 1]), (0.5, 0.8, 1.0))
    expected = pd.DataFrame(
        {
            "critical_capture": [0.0, 0.5, 1.0],
            "residual_critical_error": [0.5, 0.0, 0.0],
            "review_precision": [0.0, 0.5, 0.5],
            "critical_count": [2.0, 2.0, 2.0],
        }
    )
    pd.testing.assert_frame_equal(result[list(expected.columns)], expected, check_exact=True)


def test_absent_critical_class_retains_uninformative_intervals():
    result = risk_coverage(np.array([[0.9, 0.1], [0.2, 0.8]]), np.array([0, 1]))
    assert (result["critical_count"] == 0).all()
    assert (result["critical_capture"] == 0).all()
    assert (result["critical_capture_ci95_low"] == 0).all()
    assert (result["critical_capture_ci95_high"] == 1).all()


def test_empty_thresholds_return_no_operating_points():
    assert risk_coverage(np.array([[0.9, 0.1]]), np.array([0]), ()).empty
