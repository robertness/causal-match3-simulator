from __future__ import annotations

from match3_simulator.plot_queries import _curve_values, _select_landmark_report


def test_plot_selects_first_seed_from_validation_suite() -> None:
    report = {
        "seed_reports": [
            {"seed": 4409, "passed": True, "levels": {"orchard": {}}},
            {"seed": 5519, "passed": True, "levels": {"orchard": {}}},
        ]
    }
    selected, subtitle = _select_landmark_report(report)
    assert selected["seed"] == 4409
    assert subtitle == "validation seed 4409 shown; 2/2 seeds passed"


def test_plot_selects_engine_curves_and_bootstrap_intervals() -> None:
    values = _curve_values(
        {
            "engine": {"grid": [-1.0, 0.0, 1.0], "causal": [0.2, 0.1, 0.3]},
            "bootstrap": {
                "causal_recommendation_contrast": {
                    "estimate": 0.1,
                    "lower": 0.02,
                    "upper": 0.18,
                },
                "observational_recommendation_contrast": {
                    "estimate": -0.1,
                    "lower": -0.18,
                    "upper": -0.02,
                },
            },
        }
    )

    assert values["grid"] == [-1.0, 0.0, 1.0]
    assert values["contrast_intervals"]["causal"]["lower"] == 0.02