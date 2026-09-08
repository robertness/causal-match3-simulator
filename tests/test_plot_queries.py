from __future__ import annotations

from match3_simulator.plot_queries import _select_landmark_report


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