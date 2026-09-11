from __future__ import annotations

import json

from match3_simulator import calibrate


def test_goal_count_uses_environment_selected_wrapped_table(
    monkeypatch, tmp_path
) -> None:
    path = tmp_path / "provisional.json"
    path.write_text(
        json.dumps(
            {
                "status": "provisional",
                "table": {
                    "orchard": {
                        "curve": [
                            {"E": -1.0, "goal_count": 11},
                            {"E": 1.0, "goal_count": 31},
                        ]
                    }
                },
            }
        )
    )
    monkeypatch.setenv("MATCH3_CALIBRATION_PATH", str(path))
    calibrate._CACHE.clear()

    assert calibrate.goal_count_for_E("orchard", 0.0) == 21