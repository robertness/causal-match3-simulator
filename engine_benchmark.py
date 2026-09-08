"""Run and report board-engine landmark churn benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from .calibrate import WIN_PROPENSITY_PATH, load_win_propensity_model
from .causal_queries import (
    ASSIGNMENT_SCHEDULE,
    AssignmentSchedule,
    EngineOutcomeSurface,
    LandmarkRiskSet,
    bootstrap_engine_curves,
    compare_engine_curves,
    compare_oracle_curves,
    comparison_report,
    engine_outcome_surface,
    generate_engine_landmark_cohort,
    save_engine_outcome_surface,
)
from .retention import CHURN_SCHEDULE, ChurnSchedule, MasteryConfig
from .spec import BENCHMARK_CONFIG, BenchmarkConfig


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _subset_risk_set(risk_set: LandmarkRiskSet, rows: np.ndarray) -> LandmarkRiskSet:
    return LandmarkRiskSet(
        level_name=risk_set.level_name,
        skills=risk_set.skills[rows],
        tier_indices=risk_set.tier_indices[rows],
        mastery_before=risk_set.mastery_before[rows],
        assignment_locations=risk_set.assignment_locations[rows],
        assignment_sigma=risk_set.assignment_sigma,
        player_ids=risk_set.player_ids[rows],
        exogenous_seeds=risk_set.exogenous_seeds[rows],
    )


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def engine_level_report(
    risk_set: LandmarkRiskSet,
    surface: EngineOutcomeSurface,
    propensity_model,
    *,
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
    mastery_config: MasteryConfig = MasteryConfig(),
    n_bootstrap: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Summarize one engine curve and its surrogate calibration check."""
    grid_benchmark = replace(benchmark, e_grid=tuple(map(float, surface.grid)))
    engine = compare_engine_curves(
        risk_set,
        surface,
        mastery_config=mastery_config,
    )
    surrogate = compare_oracle_curves(
        risk_set,
        propensity_model,
        mastery_config=mastery_config,
        benchmark=grid_benchmark,
    )
    engine_report = comparison_report(engine, grid_benchmark)
    bootstrap = (
        bootstrap_engine_curves(
            risk_set,
            surface,
            mastery_config=mastery_config,
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
        if n_bootstrap
        else None
    )
    if bootstrap is not None:
        causal_interval = bootstrap["causal_recommendation_contrast"]
        observational_interval = bootstrap[
            "observational_recommendation_contrast"
        ]
        engine_report["gates"]["causal_interval_excludes_zero"] = (
            causal_interval["lower"] > 0.0
        )
        engine_report["gates"]["observational_interval_excludes_zero"] = (
            observational_interval["upper"] < 0.0
        )
        engine_report["passed"] = all(engine_report["gates"].values())
    return _json_safe(
        {
            "estimator": "board_engine",
            "passed": engine_report["passed"],
            "n_players": len(risk_set.skills),
            "rollouts_per_player": surface.outcomes.shape[2],
            "n_rollouts": surface.outcomes.size,
            "engine": engine_report,
            "bootstrap": bootstrap,
            "surrogate_check": {
                "role": "calibration_diagnostic_only",
                "causal_curve_mae": float(
                    np.mean(np.abs(engine.causal - surrogate.causal))
                ),
                "observational_curve_mae": float(
                    np.mean(
                        np.abs(engine.observational - surrogate.observational)
                    )
                ),
                "causal_optimum": surrogate.causal_optimum,
                "observational_optimum": surrogate.observational_optimum,
                "causal": surrogate.causal.tolist(),
                "observational": surrogate.observational.tolist(),
            },
        }
    )


def evaluate_engine_benchmark(
    propensity_model,
    *,
    n_players: int,
    seed: int,
    output_dir: str | Path,
    status: str = "pilot",
    max_engine_players: int | None = None,
    e_grid: tuple[float, ...] = BENCHMARK_CONFIG.e_grid,
    rollouts_per_player: int = 1,
    workers: int = 1,
    n_bootstrap: int = 0,
    assignment: AssignmentSchedule = ASSIGNMENT_SCHEDULE,
    churn_config: ChurnSchedule = CHURN_SCHEDULE,
    mastery_config: MasteryConfig = MasteryConfig(),
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
) -> dict[str, object]:
    """Run engine warm-up and target interventions for every level."""
    if status not in {"pilot", "validation"}:
        raise ValueError("status must be 'pilot' or 'validation'")
    if max_engine_players is not None and max_engine_players < 1:
        raise ValueError("max_engine_players must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cohort = generate_engine_landmark_cohort(
        propensity_model,
        n_players=n_players,
        seed=seed,
        assignment=assignment,
        churn_config=churn_config,
        mastery_config=mastery_config,
        benchmark=benchmark,
        workers=workers,
    )
    n_engine_players = min(
        cohort.n_active_players,
        max_engine_players or cohort.n_active_players,
    )
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, benchmark.landmark_attempt, 2])
    )
    rows = np.sort(
        rng.choice(
            cohort.n_active_players,
            size=n_engine_players,
            replace=False,
        )
    )
    levels: dict[str, object] = {}
    for level_index, full_risk_set in enumerate(cohort.risk_sets):
        risk_set = _subset_risk_set(full_risk_set, rows)
        surface = engine_outcome_surface(
            risk_set,
            grid=e_grid,
            rollouts_per_player=rollouts_per_player,
            workers=workers,
        )
        surface_path = save_engine_outcome_surface(
            surface, output / f"{risk_set.level_name}-outcomes.npz"
        )
        report = engine_level_report(
            risk_set,
            surface,
            propensity_model,
            benchmark=benchmark,
            mastery_config=mastery_config,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=int(
                np.random.SeedSequence([seed, level_index, 3]).generate_state(1)[0]
            ),
        )
        report["outcome_artifact"] = {
            "path": surface_path.name,
            "sha256": hashlib.sha256(surface_path.read_bytes()).hexdigest(),
        }
        levels[risk_set.level_name] = report
    configuration = {
        "seed": seed,
        "n_players": n_players,
        "max_engine_players": max_engine_players,
        "e_grid": list(e_grid),
        "rollouts_per_player": rollouts_per_player,
        "n_bootstrap": n_bootstrap,
        "benchmark": asdict(benchmark),
        "mastery": asdict(mastery_config),
        "assignment": asdict(assignment),
        "churn": asdict(churn_config),
    }
    configuration_sha256 = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()
    propensity_sha256 = hashlib.sha256(
        json.dumps(propensity_model.to_dict(), sort_keys=True).encode()
    ).hexdigest()
    document = _json_safe(
        {
            "schema_version": 1,
            "status": status,
            "estimator": "board_engine",
            "risk_set_source": "board_engine",
            "passed": all(bool(report["passed"]) for report in levels.values()),
            "code_sha": _git_revision(),
            "configuration_sha256": configuration_sha256,
            "response_surface_sha256": propensity_sha256,
            "seed": seed,
            "n_initial_players": cohort.n_initial_players,
            "n_landmark_players": cohort.n_active_players,
            "survival_fraction": cohort.survival_fraction,
            "n_engine_players": n_engine_players,
            "e_grid": list(e_grid),
            "rollouts_per_player": rollouts_per_player,
            "workers": workers,
            "n_bootstrap": n_bootstrap,
            "runtime_seconds": time.perf_counter() - started,
            "benchmark": asdict(benchmark),
            "mastery": asdict(mastery_config),
            "assignment": asdict(assignment),
            "churn": asdict(churn_config),
            "levels": levels,
        }
    )
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n"
    )
    return document


def main() -> None:  # pragma: no cover - exercised through CLI smoke runs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=int, default=64)
    parser.add_argument("--engine-players", type=int, default=None)
    parser.add_argument("--seed", type=int, default=BENCHMARK_CONFIG.calibration_seeds[0])
    parser.add_argument("--rollouts-per-player", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument(
        "--e-grid",
        type=float,
        nargs="+",
        default=list(BENCHMARK_CONFIG.e_grid),
    )
    parser.add_argument("--status", choices=("pilot", "validation"), default="pilot")
    parser.add_argument("--propensity", type=Path, default=WIN_PROPENSITY_PATH)
    parser.add_argument("--out", type=Path, default=Path("data/engine-pilot"))
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    model = load_win_propensity_model(args.propensity)
    report = evaluate_engine_benchmark(
        model,
        n_players=args.players,
        max_engine_players=args.engine_players,
        seed=args.seed,
        output_dir=args.out,
        status=args.status,
        e_grid=tuple(args.e_grid),
        rollouts_per_player=args.rollouts_per_player,
        workers=args.workers,
        n_bootstrap=args.bootstrap,
    )
    print(
        f"wrote {args.out / 'report.json'}  "
        f"players={report['n_engine_players']}  passed={report['passed']}"
    )
    if args.require_pass and not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()