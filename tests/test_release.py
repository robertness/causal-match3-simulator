from __future__ import annotations

import json
import math

from match3_simulator.release import (
    generate_release,
    load_accepted_spec,
)


def test_randomized_assignment_preserves_marginal_variance() -> None:
    spec = load_accepted_spec()
    natural = spec["regimes"]["natural"]
    randomized = spec["regimes"]["randomized"]

    assert randomized["skill_gains"] == [0.0, 0.0, 0.0]
    assert randomized["sigmas"] == [
        math.hypot(gain, sigma)
        for gain, sigma in zip(
            natural["skill_gains"], natural["sigmas"], strict=True
        )
    ]


def test_release_smoke_is_atomic_disjoint_and_audited(tmp_path) -> None:
    output = tmp_path / "release"
    manifest_path = generate_release(
        output,
        players=6,
        max_attempts=1,
        shard_size=3,
        workers=1,
        include_transitions=False,
    )
    manifest = json.loads(manifest_path.read_text())
    qc = json.loads((output / "qc.json").read_text())

    assert not output.with_name("release.partial").exists()
    assert manifest["status"] == "accepted_reference_dataset"
    assert qc["status"] == "pass"
    assert qc["split_counts"] == {
        "test": 1,
        "train": 4,
        "validation": 1,
    }
    assert qc["player_disjoint_splits"]
    assert qc["matched_player_skills"]
    assert set(manifest["regimes"]) == {"natural", "randomized"}
    for regime in manifest["regimes"].values():
        assert regime["counts"]["players"] == 6
        assert len(regime["shards"]) == 2