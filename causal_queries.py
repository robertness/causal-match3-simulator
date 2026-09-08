"""Oracle observational and interventional landmark churn queries."""

from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .retention import (
    CHURN_SCHEDULE,
    ChurnConfig,
    ChurnSchedule,
    MasteryConfig,
    WinPropensityModel,
    expected_mastery_churn,
    mastery_mismatch_hazard,
    update_mastery,
)
from .spec import (
    BENCHMARK_CONFIG,
    BenchmarkConfig,
    Difficulty,
    LevelContext,
    PlayerSkill,
    SKILL_COVARIANCE,
    SKILL_NAMES,
)


@dataclass(frozen=True)
class AssignmentConfig:
    """Natural DDA assignment parameters used by landmark simulations."""

    skill_gain: float = 0.8
    sigma: float = 0.65

    def __post_init__(self) -> None:
        if self.skill_gain < 0:
            raise ValueError("skill_gain must be non-negative")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")


@dataclass(frozen=True)
class AssignmentSchedule:
    """Validated level-specific natural DDA assignment parameters."""

    level_names: tuple[str, ...] = ("orchard", "harbour", "foundry")
    skill_gains: tuple[float, ...] = (4.0, 4.0, 8.0)
    sigmas: tuple[float, ...] = (1.2, 1.2, 2.4)

    def __post_init__(self) -> None:
        n_levels = len(self.level_names)
        if len(set(self.level_names)) != n_levels:
            raise ValueError("level_names must be unique")
        if len(self.skill_gains) != n_levels or len(self.sigmas) != n_levels:
            raise ValueError("assignment parameters must align with levels")
        if any(value < 0 for value in self.skill_gains):
            raise ValueError("skill gains must be non-negative")
        if any(value <= 0 for value in self.sigmas):
            raise ValueError("sigmas must be positive")

    def for_level(self, level_name: str) -> AssignmentConfig:
        index = self.level_names.index(level_name)
        return AssignmentConfig(self.skill_gains[index], self.sigmas[index])


ASSIGNMENT_SCHEDULE = AssignmentSchedule()


def _assignment_for_level(
    assignment: AssignmentConfig | AssignmentSchedule, level_name: str
) -> AssignmentConfig:
    return (
        assignment.for_level(level_name)
        if isinstance(assignment, AssignmentSchedule)
        else assignment
    )


def _churn_for_level(
    churn: ChurnConfig | ChurnSchedule, level_name: str
) -> ChurnConfig:
    return churn.for_level(level_name) if isinstance(churn, ChurnSchedule) else churn


@dataclass(frozen=True)
class LandmarkRiskSet:
    """Pre-treatment records for players active at the landmark attempt."""

    level_name: str
    skills: np.ndarray
    tier_indices: np.ndarray
    mastery_before: np.ndarray
    assignment_locations: np.ndarray
    assignment_sigma: float
    player_ids: np.ndarray | None = None
    exogenous_seeds: np.ndarray | None = None

    def __post_init__(self) -> None:
        skills = np.asarray(self.skills)
        tiers = np.asarray(self.tier_indices)
        mastery = np.asarray(self.mastery_before)
        locations = np.asarray(self.assignment_locations)
        if skills.ndim != 2 or skills.shape[1] != len(SKILL_NAMES):
            raise ValueError("skills must have shape (players, skills)")
        if tiers.shape != (skills.shape[0],):
            raise ValueError("tier_indices must align with skills")
        if mastery.shape != (skills.shape[0],):
            raise ValueError("mastery_before must align with skills")
        if np.any((mastery < 0.0) | (mastery > 1.0)):
            raise ValueError("mastery_before must lie in [0, 1]")
        if locations.shape != (skills.shape[0],):
            raise ValueError("assignment_locations must align with skills")
        if skills.shape[0] == 0:
            raise ValueError("risk set cannot be empty")
        if self.assignment_sigma <= 0:
            raise ValueError("assignment_sigma must be positive")
        player_ids = (
            np.arange(skills.shape[0], dtype=np.int64)
            if self.player_ids is None
            else np.asarray(self.player_ids, dtype=np.int64)
        )
        seeds = (
            player_ids.astype(np.uint32)
            if self.exogenous_seeds is None
            else np.asarray(self.exogenous_seeds, dtype=np.uint32)
        )
        if player_ids.shape != (skills.shape[0],):
            raise ValueError("player_ids must align with skills")
        if len(np.unique(player_ids)) != len(player_ids):
            raise ValueError("player_ids must be unique")
        if seeds.shape != (skills.shape[0],):
            raise ValueError("exogenous_seeds must align with skills")
        object.__setattr__(self, "player_ids", player_ids)
        object.__setattr__(self, "exogenous_seeds", seeds)


@dataclass(frozen=True)
class EngineOutcomeSurface:
    """Paired target-attempt outcomes indexed by E, player, and replicate."""

    level_name: str
    grid: np.ndarray
    player_ids: np.ndarray
    rollout_seeds: np.ndarray
    outcomes: np.ndarray

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid, dtype=np.float64)
        player_ids = np.asarray(self.player_ids, dtype=np.int64)
        seeds = np.asarray(self.rollout_seeds, dtype=np.uint32)
        outcomes = np.asarray(self.outcomes, dtype=np.int8)
        if grid.ndim != 1 or len(grid) == 0:
            raise ValueError("grid must be a non-empty vector")
        if player_ids.ndim != 1 or len(player_ids) == 0:
            raise ValueError("player_ids must be a non-empty vector")
        if seeds.ndim != 2 or seeds.shape[0] != len(player_ids):
            raise ValueError("rollout_seeds must have shape (players, replicates)")
        if outcomes.shape != (len(grid), *seeds.shape):
            raise ValueError("outcomes must have shape (grid, players, replicates)")
        if np.any((outcomes != 0) & (outcomes != 1)):
            raise ValueError("outcomes must be binary")
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "player_ids", player_ids)
        object.__setattr__(self, "rollout_seeds", seeds)
        object.__setattr__(self, "outcomes", outcomes)


@dataclass(frozen=True)
class LandmarkCohort:
    """One eligible landmark population cloned across all level contexts."""

    n_initial_players: int
    active_player_ids: np.ndarray
    risk_sets: tuple[LandmarkRiskSet, ...]

    @property
    def n_active_players(self) -> int:
        return len(self.active_player_ids)

    @property
    def survival_fraction(self) -> float:
        return self.n_active_players / self.n_initial_players


@dataclass(frozen=True)
class OverlapDiagnostic:
    e: float
    effective_sample_size: float
    effective_sample_fraction: float
    maximum_normalized_weight: float


@dataclass(frozen=True)
class CurveComparison:
    grid: np.ndarray
    causal: np.ndarray
    observational: np.ndarray
    causal_optimum: float
    observational_optimum: float
    causal_recommendation_contrast: float
    observational_recommendation_contrast: float
    causal_left_contrast: float
    causal_right_contrast: float
    overlap: tuple[OverlapDiagnostic, ...]

    @property
    def recommendation_gap(self) -> float:
        return abs(self.causal_optimum - self.observational_optimum)


@dataclass(frozen=True)
class ContrastInterval:
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class RecommendationContrastIntervals:
    causal: ContrastInterval
    observational: ContrastInterval


def _normalized_assignment_weights(
    risk_set: LandmarkRiskSet, served_difficulty: float
) -> np.ndarray:
    residual = (
        served_difficulty - np.asarray(risk_set.assignment_locations)
    ) / risk_set.assignment_sigma
    log_weights = -0.5 * residual**2
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return weights / np.sum(weights)


def _effective_skills(skills: np.ndarray, level: LevelContext) -> np.ndarray:
    demand = level.demand_weights()
    scale = float(np.sqrt(demand @ SKILL_COVARIANCE @ demand))
    return np.asarray(skills, dtype=np.float64) @ demand / scale


def _engine_warmup_summary(
    payload: tuple[
        WinPropensityModel,
        int,
        int,
        BenchmarkConfig,
        ChurnConfig | ChurnSchedule,
        MasteryConfig,
    ],
) -> tuple[int, np.ndarray, float, bool]:
    """Run one player's pre-landmark engine history and retain causal state."""
    propensity_model, player_id, seed, benchmark, churn_config, mastery_config = (
        payload
    )
    from .retention import simulate_player_trajectory
    from .scm import sample_K

    if benchmark.landmark_attempt == 1:
        import pyro

        skill_seed = int(
            np.random.SeedSequence([seed, player_id, 0]).generate_state(1)[0]
        )
        pyro.set_rng_seed(skill_seed)
        player = sample_K()
        return player_id, player.as_array(), mastery_config.initial, True
    trajectory = simulate_player_trajectory(
        propensity_model,
        player_id=player_id,
        seed=seed,
        max_attempts=benchmark.landmark_attempt - 1,
        benchmark=benchmark,
        churn_config=churn_config,
        mastery_config=mastery_config,
    )
    active = (
        len(trajectory.attempts) == benchmark.landmark_attempt - 1
        and not trajectory.churned
    )
    mastery_before = (
        trajectory.attempts[-1].mastery_after
        if trajectory.attempts
        else mastery_config.initial
    )
    return player_id, trajectory.player.as_array(), mastery_before, active


def generate_engine_landmark_cohort(
    propensity_model: WinPropensityModel,
    *,
    n_players: int,
    seed: int,
    assignment: AssignmentSchedule = ASSIGNMENT_SCHEDULE,
    churn_config: ChurnConfig | ChurnSchedule = CHURN_SCHEDULE,
    mastery_config: MasteryConfig = MasteryConfig(),
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
    workers: int = 1,
) -> LandmarkCohort:
    """Generate pre-landmark mastery from actual board-engine trajectories."""
    if n_players < 1:
        raise ValueError("n_players must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if assignment != ASSIGNMENT_SCHEDULE:
        raise ValueError("engine warm-up requires the simulator assignment schedule")
    from concurrent.futures import ProcessPoolExecutor

    from .scm import LEVELS, TIER_LOGITS, TIER_NAMES, TIER_PROBS

    tasks = [
        (
            propensity_model,
            player_id,
            seed,
            benchmark,
            churn_config,
            mastery_config,
        )
        for player_id in range(n_players)
    ]
    if workers == 1:
        summaries = [_engine_warmup_summary(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            summaries = list(
                executor.map(
                    _engine_warmup_summary,
                    tasks,
                    chunksize=max(1, len(tasks) // (workers * 4)),
                )
            )
    active_summaries = [summary for summary in summaries if summary[3]]
    if not active_summaries:
        raise RuntimeError("no players survived to the landmark attempt")
    active_player_ids = np.asarray(
        [summary[0] for summary in active_summaries], dtype=np.int64
    )
    active_skills = np.asarray(
        [summary[1] for summary in active_summaries], dtype=np.float64
    )
    active_mastery = np.asarray(
        [summary[2] for summary in active_summaries], dtype=np.float64
    )
    target_rng = np.random.default_rng(
        np.random.SeedSequence([seed, benchmark.landmark_attempt, 1])
    )
    tier_logits = np.asarray(TIER_LOGITS, dtype=np.float64)
    exogenous_seeds = np.asarray(
        [
            np.random.SeedSequence(
                [seed, int(player_id), benchmark.landmark_attempt]
            ).generate_state(1)[0]
            for player_id in active_player_ids
        ],
        dtype=np.uint32,
    )
    risk_sets = []
    for level in LEVELS:
        level_assignment = assignment.for_level(level.name)
        tiers = target_rng.choice(
            len(TIER_NAMES), size=len(active_skills), p=TIER_PROBS
        )
        locations = (
            tier_logits[tiers]
            + level_assignment.skill_gain * _effective_skills(active_skills, level)
        )
        risk_sets.append(
            LandmarkRiskSet(
                level_name=level.name,
                skills=active_skills.copy(),
                tier_indices=tiers,
                mastery_before=active_mastery.copy(),
                assignment_locations=locations,
                assignment_sigma=level_assignment.sigma,
                player_ids=active_player_ids.copy(),
                exogenous_seeds=exogenous_seeds.copy(),
            )
        )
    return LandmarkCohort(
        n_initial_players=n_players,
        active_player_ids=active_player_ids,
        risk_sets=tuple(risk_sets),
    )


def generate_landmark_cohort(
    propensity_model: WinPropensityModel,
    *,
    n_players: int,
    seed: int,
    assignment: AssignmentConfig | AssignmentSchedule = ASSIGNMENT_SCHEDULE,
    churn_config: ChurnConfig | ChurnSchedule = CHURN_SCHEDULE,
    mastery_config: MasteryConfig = MasteryConfig(),
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
) -> LandmarkCohort:
    """Generate the warm-up risk set, then clone it across landmark levels.

    The primary benchmark uses an observation-only warm-up, so every player is
    eligible at the landmark. ``warmup_churn_scale > 0`` introduces absorbing
    pre-landmark attrition as a survivor-selection sensitivity analysis.
    """
    if n_players < 1:
        raise ValueError("n_players must be positive")
    from .scm import (
        LEVEL_PROBS,
        LEVELS,
        TIER_LOGITS,
        TIER_NAMES,
        TIER_PROBS,
    )

    level_names = tuple(level.name for level in LEVELS)
    if propensity_model.level_names != level_names:
        raise ValueError("propensity model levels do not match simulator levels")
    if propensity_model.tier_names != TIER_NAMES:
        raise ValueError("propensity model tiers do not match simulator tiers")

    rng = np.random.default_rng(seed)
    skills = rng.multivariate_normal(
        np.zeros(len(SKILL_NAMES)), SKILL_COVARIANCE, size=n_players
    )
    effective_by_level = np.column_stack(
        [_effective_skills(skills, level) for level in LEVELS]
    )
    active = np.ones(n_players, dtype=bool)
    mastery = np.full(n_players, mastery_config.initial, dtype=np.float64)
    tier_logits = np.asarray(TIER_LOGITS, dtype=np.float64)

    gains = np.asarray(
        [
            _assignment_for_level(assignment, level.name).skill_gain
            for level in LEVELS
        ]
    )
    sigmas = np.asarray(
        [
            _assignment_for_level(assignment, level.name).sigma
            for level in LEVELS
        ]
    )
    for _ in range(1, benchmark.landmark_attempt):
        level_indices = rng.choice(len(LEVELS), size=n_players, p=LEVEL_PROBS)
        tier_indices = rng.choice(len(TIER_NAMES), size=n_players, p=TIER_PROBS)
        assignment_locations = (
            tier_logits[tier_indices]
            + gains[level_indices]
            * effective_by_level[np.arange(n_players), level_indices]
        )
        served = assignment_locations + rng.normal(size=n_players) * sigmas[
            level_indices
        ]
        win_probability = np.empty(n_players, dtype=np.float64)
        hazard = np.empty(n_players, dtype=np.float64)
        for level_index, level in enumerate(LEVELS):
            mask = level_indices == level_index
            if np.any(mask):
                win_probability[mask] = propensity_model.probabilities(
                    level.name,
                    tier_indices[mask],
                    skills[mask],
                    served[mask],
                )
        outcomes = rng.binomial(1, win_probability)
        mastery_after = update_mastery(mastery, outcomes, mastery_config)
        if benchmark.warmup_churn_scale > 0.0:
            for level_index, level in enumerate(LEVELS):
                mask = level_indices == level_index
                if np.any(mask):
                    hazard[mask] = mastery_mismatch_hazard(
                        mastery_after[mask],
                        _churn_for_level(churn_config, level.name),
                    )
            churned = (
                rng.random(n_players)
                < benchmark.warmup_churn_scale * hazard
            )
            active &= ~churned
        mastery = mastery_after

    active_player_ids = np.flatnonzero(active)
    if len(active_player_ids) == 0:
        raise RuntimeError("no players survived to the landmark attempt")
    active_skills = skills[active]
    exogenous_seeds = np.asarray(
        [
            np.random.SeedSequence(
                [seed, int(player_id), benchmark.landmark_attempt]
            ).generate_state(1)[0]
            for player_id in active_player_ids
        ],
        dtype=np.uint32,
    )
    risk_sets: list[LandmarkRiskSet] = []
    for level_index, level in enumerate(LEVELS):
        level_assignment = _assignment_for_level(assignment, level.name)
        tiers = rng.choice(len(TIER_NAMES), size=len(active_skills), p=TIER_PROBS)
        locations = (
            tier_logits[tiers]
            + level_assignment.skill_gain
            * effective_by_level[active, level_index]
        )
        risk_sets.append(
            LandmarkRiskSet(
                level_name=level.name,
                skills=active_skills.copy(),
                tier_indices=tiers,
                mastery_before=mastery[active].copy(),
                assignment_locations=locations,
                assignment_sigma=level_assignment.sigma,
                player_ids=active_player_ids.copy(),
                exogenous_seeds=exogenous_seeds.copy(),
            )
        )
    return LandmarkCohort(
        n_initial_players=n_players,
        active_player_ids=active_player_ids,
        risk_sets=tuple(risk_sets),
    )


def randomized_assignment_control(
    cohort: LandmarkCohort,
    *,
    sigma: float = 1.25,
) -> LandmarkCohort:
    """Replace natural assignment by E independent of K and D within level."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    risk_sets = tuple(
        LandmarkRiskSet(
            level_name=risk_set.level_name,
            skills=risk_set.skills,
            tier_indices=risk_set.tier_indices,
            mastery_before=risk_set.mastery_before,
            assignment_locations=np.zeros(len(risk_set.skills)),
            assignment_sigma=sigma,
            player_ids=risk_set.player_ids,
            exogenous_seeds=risk_set.exogenous_seeds,
        )
        for risk_set in cohort.risk_sets
    )
    return LandmarkCohort(
        n_initial_players=cohort.n_initial_players,
        active_player_ids=cohort.active_player_ids.copy(),
        risk_sets=risk_sets,
    )


def overlap_diagnostic(
    risk_set: LandmarkRiskSet, served_difficulty: float
) -> OverlapDiagnostic:
    weights = _normalized_assignment_weights(risk_set, served_difficulty)
    effective_size = float(1.0 / np.sum(weights**2))
    return OverlapDiagnostic(
        e=float(served_difficulty),
        effective_sample_size=effective_size,
        effective_sample_fraction=effective_size / len(weights),
        maximum_normalized_weight=float(np.max(weights)),
    )


def engine_outcome_surface(
    risk_set: LandmarkRiskSet,
    *,
    grid: np.ndarray | tuple[float, ...] = BENCHMARK_CONFIG.e_grid,
    rollouts_per_player: int = 2,
    workers: int = 1,
) -> EngineOutcomeSurface:
    """Run paired target-attempt interventions through the board engine."""
    if rollouts_per_player < 1:
        raise ValueError("rollouts_per_player must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    from concurrent.futures import ProcessPoolExecutor

    from .calibrate import _run_episode_tasks, goal_count_for_E
    from .scm import (
        DEFAULT_GOAL_COLOUR,
        LEVELS,
        TIER_LOGITS,
        TIER_MOVE_BUDGETS,
    )

    grid_array = np.asarray(grid, dtype=np.float64)
    if grid_array.ndim != 1 or len(grid_array) == 0:
        raise ValueError("grid must be a non-empty vector")
    level = next(
        (candidate for candidate in LEVELS if candidate.name == risk_set.level_name),
        None,
    )
    if level is None:
        raise ValueError(f"unknown level {risk_set.level_name!r}")
    rollout_seeds = np.asarray(
        [
            [
                np.random.SeedSequence([int(base_seed), replicate]).generate_state(1)[0]
                for replicate in range(rollouts_per_player)
            ]
            for base_seed in risk_set.exogenous_seeds
        ],
        dtype=np.uint32,
    )
    players = [
        PlayerSkill(tuple(map(float, values))) for values in risk_set.skills
    ]
    difficulties = [
        Difficulty(
            move_budget=TIER_MOVE_BUDGETS[int(tier)],
            goal_colour=DEFAULT_GOAL_COLOUR,
            goal_count=goal_count_for_E(level.name, TIER_LOGITS[int(tier)]),
            baseline=TIER_LOGITS[int(tier)],
        )
        for tier in risk_set.tier_indices
    ]
    tasks = [
        (
            level,
            difficulties[player_index],
            players[player_index],
            float(served_difficulty),
            None,
            int(rollout_seeds[player_index, replicate]),
        )
        for served_difficulty in grid_array
        for player_index in range(len(players))
        for replicate in range(rollouts_per_player)
    ]
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        outcomes = _run_episode_tasks(tasks, executor)
    finally:
        if executor is not None:
            executor.shutdown()
    return EngineOutcomeSurface(
        level_name=risk_set.level_name,
        grid=grid_array,
        player_ids=risk_set.player_ids,
        rollout_seeds=rollout_seeds,
        outcomes=np.asarray(outcomes, dtype=np.int8).reshape(
            len(grid_array), len(players), rollouts_per_player
        ),
    )


def save_engine_outcome_surface(
    surface: EngineOutcomeSurface, path: str | Path
) -> Path:
    """Persist paired engine outcomes without serializing Python objects."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray([1], dtype=np.int16),
        level_name=np.asarray(surface.level_name),
        grid=surface.grid,
        player_ids=surface.player_ids,
        rollout_seeds=surface.rollout_seeds,
        outcomes=surface.outcomes,
    )
    return output


def load_engine_outcome_surface(path: str | Path) -> EngineOutcomeSurface:
    """Load and validate a paired engine outcome artifact."""
    with np.load(Path(path), allow_pickle=False) as values:
        version = int(values["schema_version"][0])
        if version != 1:
            raise ValueError(f"unsupported engine outcome schema {version}")
        return EngineOutcomeSurface(
            level_name=str(values["level_name"].item()),
            grid=values["grid"],
            player_ids=values["player_ids"],
            rollout_seeds=values["rollout_seeds"],
            outcomes=values["outcomes"],
        )


def _engine_player_hazards(
    risk_set: LandmarkRiskSet,
    surface: EngineOutcomeSurface,
    churn_config: ChurnConfig,
    mastery_config: MasteryConfig,
) -> np.ndarray:
    mastery_before = np.asarray(risk_set.mastery_before)[None, :, None]
    mastery_after = update_mastery(
        mastery_before, surface.outcomes, mastery_config
    )
    return mastery_mismatch_hazard(
        mastery_after, churn_config
    ).mean(axis=2)


def compare_engine_curves(
    risk_set: LandmarkRiskSet,
    surface: EngineOutcomeSurface,
    *,
    churn_config: ChurnConfig | None = None,
    mastery_config: MasteryConfig = MasteryConfig(),
) -> CurveComparison:
    """Compare standardized and assignment-weighted engine churn curves."""
    if surface.level_name != risk_set.level_name:
        raise ValueError("surface level does not match risk set")
    if not np.array_equal(surface.player_ids, risk_set.player_ids):
        raise ValueError("surface players do not match risk set")
    current_churn = churn_config or CHURN_SCHEDULE.for_level(risk_set.level_name)
    player_hazards = _engine_player_hazards(
        risk_set, surface, current_churn, mastery_config
    )
    causal = player_hazards.mean(axis=1)
    observational = np.asarray(
        [
            _normalized_assignment_weights(risk_set, float(e)) @ hazards
            for e, hazards in zip(surface.grid, player_hazards)
        ]
    )
    grid = surface.grid
    causal_optimum = select_grid_optimum(grid, causal)
    observational_optimum = select_grid_optimum(grid, observational)
    causal_index = int(np.flatnonzero(grid == causal_optimum)[0])
    observational_index = int(np.flatnonzero(grid == observational_optimum)[0])
    left = np.flatnonzero(np.isclose(grid, causal_optimum - 1.0))
    right = np.flatnonzero(np.isclose(grid, causal_optimum + 1.0))
    return CurveComparison(
        grid=grid,
        causal=causal,
        observational=observational,
        causal_optimum=causal_optimum,
        observational_optimum=observational_optimum,
        causal_recommendation_contrast=float(
            causal[observational_index] - causal[causal_index]
        ),
        observational_recommendation_contrast=float(
            observational[observational_index] - observational[causal_index]
        ),
        causal_left_contrast=(
            float(causal[int(left[0])] - causal[causal_index])
            if len(left)
            else -np.inf
        ),
        causal_right_contrast=(
            float(causal[int(right[0])] - causal[causal_index])
            if len(right)
            else -np.inf
        ),
        overlap=tuple(overlap_diagnostic(risk_set, float(e)) for e in grid),
    )


def bootstrap_engine_curves(
    risk_set: LandmarkRiskSet,
    surface: EngineOutcomeSurface,
    *,
    churn_config: ChurnConfig | None = None,
    mastery_config: MasteryConfig = MasteryConfig(),
    n_bootstrap: int = 500,
    seed: int = 0,
) -> dict[str, object]:
    """Resample whole players while preserving paired E and rollout outcomes."""
    if n_bootstrap < 20:
        raise ValueError("n_bootstrap must be at least 20")
    comparison = compare_engine_curves(
        risk_set,
        surface,
        churn_config=churn_config,
        mastery_config=mastery_config,
    )
    current_churn = churn_config or CHURN_SCHEDULE.for_level(risk_set.level_name)
    player_hazards = _engine_player_hazards(
        risk_set, surface, current_churn, mastery_config
    )
    residual = (
        surface.grid[:, None] - np.asarray(risk_set.assignment_locations)[None, :]
    ) / risk_set.assignment_sigma
    log_weights = -0.5 * residual**2
    log_weights -= np.max(log_weights, axis=1, keepdims=True)
    raw_weights = np.exp(log_weights)
    n_players = player_hazards.shape[1]
    causal_curves = np.empty((n_bootstrap, len(surface.grid)), dtype=np.float64)
    observational_curves = np.empty_like(causal_curves)
    causal_contrasts = np.empty(n_bootstrap, dtype=np.float64)
    observational_contrasts = np.empty(n_bootstrap, dtype=np.float64)
    causal_optima = np.empty(n_bootstrap, dtype=np.float64)
    observational_optima = np.empty(n_bootstrap, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for bootstrap_index in range(n_bootstrap):
        rows = rng.integers(0, n_players, size=n_players)
        causal = player_hazards[:, rows].mean(axis=1)
        selected_weights = raw_weights[:, rows]
        observational = np.sum(
            selected_weights * player_hazards[:, rows], axis=1
        ) / np.sum(selected_weights, axis=1)
        causal_curves[bootstrap_index] = causal
        observational_curves[bootstrap_index] = observational
        causal_optimum = select_grid_optimum(surface.grid, causal)
        observational_optimum = select_grid_optimum(surface.grid, observational)
        causal_index = int(np.flatnonzero(surface.grid == causal_optimum)[0])
        observational_index = int(
            np.flatnonzero(surface.grid == observational_optimum)[0]
        )
        causal_optima[bootstrap_index] = causal_optimum
        observational_optima[bootstrap_index] = observational_optimum
        causal_contrasts[bootstrap_index] = (
            causal[observational_index] - causal[causal_index]
        )
        observational_contrasts[bootstrap_index] = (
            observational[observational_index]
            - observational[causal_index]
        )

    def interval(estimate: float, values: np.ndarray) -> dict[str, float]:
        bounds = np.quantile(values, [0.025, 0.975])
        return {
            "estimate": estimate,
            "lower": float(bounds[0]),
            "upper": float(bounds[1]),
        }

    def pointwise(values: np.ndarray) -> dict[str, list[float]]:
        bounds = np.quantile(values, [0.025, 0.975], axis=0)
        return {
            "lower": bounds[0].tolist(),
            "upper": bounds[1].tolist(),
        }

    def frequencies(values: np.ndarray) -> dict[str, float]:
        return {
            str(float(e)): float(np.mean(values == e))
            for e in surface.grid
            if np.any(values == e)
        }

    return {
        "schema_version": 1,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "resampling_unit": "player",
        "causal_pointwise": pointwise(causal_curves),
        "observational_pointwise": pointwise(observational_curves),
        "causal_recommendation_contrast": interval(
            comparison.causal_recommendation_contrast, causal_contrasts
        ),
        "observational_recommendation_contrast": interval(
            comparison.observational_recommendation_contrast,
            observational_contrasts,
        ),
        "causal_optimum_frequencies": frequencies(causal_optima),
        "observational_optimum_frequencies": frequencies(observational_optima),
    }


def oracle_causal_curve(
    risk_set: LandmarkRiskSet,
    propensity_model: WinPropensityModel,
    churn_config: ChurnConfig,
    grid: np.ndarray,
    mastery_config: MasteryConfig = MasteryConfig(),
) -> np.ndarray:
    skills = np.asarray(risk_set.skills, dtype=np.float64)
    tiers = np.asarray(risk_set.tier_indices, dtype=np.int64)
    return np.asarray(
        [
            np.mean(
                expected_mastery_churn(
                    propensity_model.probabilities(
                        risk_set.level_name, tiers, skills, float(e)
                    ),
                    risk_set.mastery_before,
                    mastery_config=mastery_config,
                    churn_config=churn_config,
                )
            )
            for e in grid
        ]
    )


def oracle_observational_curve(
    risk_set: LandmarkRiskSet,
    propensity_model: WinPropensityModel,
    churn_config: ChurnConfig,
    grid: np.ndarray,
    mastery_config: MasteryConfig = MasteryConfig(),
) -> np.ndarray:
    skills = np.asarray(risk_set.skills, dtype=np.float64)
    tiers = np.asarray(risk_set.tier_indices, dtype=np.int64)
    values = []
    for e in grid:
        hazard = expected_mastery_churn(
            propensity_model.probabilities(
                risk_set.level_name, tiers, skills, float(e)
            ),
            risk_set.mastery_before,
            mastery_config=mastery_config,
            churn_config=churn_config,
        )
        weights = _normalized_assignment_weights(risk_set, float(e))
        values.append(float(weights @ hazard))
    return np.asarray(values)


def select_grid_optimum(grid: np.ndarray, values: np.ndarray) -> float:
    grid_array = np.asarray(grid, dtype=np.float64)
    value_array = np.asarray(values, dtype=np.float64)
    if grid_array.ndim != 1 or value_array.shape != grid_array.shape:
        raise ValueError("grid and values must be aligned vectors")
    candidates = grid_array[np.isclose(value_array, np.min(value_array), atol=1e-12)]
    order = np.lexsort((candidates, np.abs(candidates)))
    return float(candidates[order[0]])


def compare_oracle_curves(
    risk_set: LandmarkRiskSet,
    propensity_model: WinPropensityModel,
    churn_config: ChurnConfig | None = None,
    mastery_config: MasteryConfig = MasteryConfig(),
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
) -> CurveComparison:
    if churn_config is None:
        churn_config = CHURN_SCHEDULE.for_level(risk_set.level_name)
    grid = np.asarray(benchmark.e_grid, dtype=np.float64)
    causal = oracle_causal_curve(
        risk_set, propensity_model, churn_config, grid, mastery_config
    )
    observational = oracle_observational_curve(
        risk_set, propensity_model, churn_config, grid, mastery_config
    )
    causal_optimum = select_grid_optimum(grid, causal)
    observational_optimum = select_grid_optimum(grid, observational)
    causal_index = int(np.flatnonzero(grid == causal_optimum)[0])
    observational_index = int(np.flatnonzero(grid == observational_optimum)[0])
    left = np.flatnonzero(np.isclose(grid, causal_optimum - 1.0))
    right = np.flatnonzero(np.isclose(grid, causal_optimum + 1.0))
    left_contrast = (
        float(causal[int(left[0])] - causal[causal_index]) if len(left) else -np.inf
    )
    right_contrast = (
        float(causal[int(right[0])] - causal[causal_index]) if len(right) else -np.inf
    )
    overlap = tuple(overlap_diagnostic(risk_set, float(e)) for e in grid)
    return CurveComparison(
        grid=grid,
        causal=causal,
        observational=observational,
        causal_optimum=causal_optimum,
        observational_optimum=observational_optimum,
        causal_recommendation_contrast=float(
            causal[observational_index] - causal[causal_index]
        ),
        observational_recommendation_contrast=float(
            observational[observational_index] - observational[causal_index]
        ),
        causal_left_contrast=left_contrast,
        causal_right_contrast=right_contrast,
        overlap=overlap,
    )


def passes_initial_gates(
    comparison: CurveComparison,
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
) -> bool:
    overlap_ok = all(
        diagnostic.effective_sample_fraction >= benchmark.minimum_ess_fraction
        and diagnostic.maximum_normalized_weight
        <= benchmark.maximum_normalized_weight
        for diagnostic in comparison.overlap
    )
    return bool(
        comparison.recommendation_gap >= benchmark.minimum_recommendation_gap
        and comparison.causal_recommendation_contrast
        >= benchmark.minimum_churn_contrast
        and comparison.observational_recommendation_contrast
        <= -benchmark.minimum_churn_contrast
        and comparison.causal_left_contrast >= benchmark.minimum_shoulder_contrast
        and comparison.causal_right_contrast >= benchmark.minimum_shoulder_contrast
        and overlap_ok
    )


def bootstrap_recommendation_contrasts(
    risk_set: LandmarkRiskSet,
    propensity_model: WinPropensityModel,
    comparison: CurveComparison,
    *,
    churn_config: ChurnConfig | None = None,
    mastery_config: MasteryConfig = MasteryConfig(),
    n_bootstrap: int = 500,
    seed: int = 0,
) -> RecommendationContrastIntervals:
    """Bootstrap fixed-optimum contrasts by resampling landmark players."""
    if n_bootstrap < 20:
        raise ValueError("n_bootstrap must be at least 20")
    if churn_config is None:
        churn_config = CHURN_SCHEDULE.for_level(risk_set.level_name)
    skills = np.asarray(risk_set.skills, dtype=np.float64)
    tiers = np.asarray(risk_set.tier_indices, dtype=np.int64)
    causal_e = comparison.causal_optimum
    observational_e = comparison.observational_optimum
    hazard_causal_e = expected_mastery_churn(
        propensity_model.probabilities(
            risk_set.level_name, tiers, skills, causal_e
        ),
        risk_set.mastery_before,
        mastery_config=mastery_config,
        churn_config=churn_config,
    )
    hazard_observational_e = expected_mastery_churn(
        propensity_model.probabilities(
            risk_set.level_name, tiers, skills, observational_e
        ),
        risk_set.mastery_before,
        mastery_config=mastery_config,
        churn_config=churn_config,
    )
    locations = np.asarray(risk_set.assignment_locations, dtype=np.float64)
    weight_causal_e = np.exp(
        -0.5 * ((causal_e - locations) / risk_set.assignment_sigma) ** 2
    )
    weight_observational_e = np.exp(
        -0.5
        * ((observational_e - locations) / risk_set.assignment_sigma) ** 2
    )

    rng = np.random.default_rng(seed)
    causal_samples = np.empty(n_bootstrap, dtype=np.float64)
    observational_samples = np.empty(n_bootstrap, dtype=np.float64)
    for bootstrap_index in range(n_bootstrap):
        rows = rng.integers(0, len(skills), size=len(skills))
        causal_samples[bootstrap_index] = np.mean(
            hazard_observational_e[rows] - hazard_causal_e[rows]
        )
        observational_samples[bootstrap_index] = (
            np.sum(
                weight_observational_e[rows] * hazard_observational_e[rows]
            )
            / np.sum(weight_observational_e[rows])
            - np.sum(weight_causal_e[rows] * hazard_causal_e[rows])
            / np.sum(weight_causal_e[rows])
        )

    causal_bounds = np.quantile(causal_samples, [0.025, 0.975])
    observational_bounds = np.quantile(observational_samples, [0.025, 0.975])
    return RecommendationContrastIntervals(
        causal=ContrastInterval(
            estimate=comparison.causal_recommendation_contrast,
            lower=float(causal_bounds[0]),
            upper=float(causal_bounds[1]),
        ),
        observational=ContrastInterval(
            estimate=comparison.observational_recommendation_contrast,
            lower=float(observational_bounds[0]),
            upper=float(observational_bounds[1]),
        ),
    )


def comparison_report(
    comparison: CurveComparison,
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
    intervals: RecommendationContrastIntervals | None = None,
) -> dict[str, object]:
    """Serialize one level's curves, recommendations, and gate results."""
    overlap_passes = [
        diagnostic.effective_sample_fraction >= benchmark.minimum_ess_fraction
        and diagnostic.maximum_normalized_weight
        <= benchmark.maximum_normalized_weight
        for diagnostic in comparison.overlap
    ]
    gates = {
        "recommendation_gap": (
            comparison.recommendation_gap >= benchmark.minimum_recommendation_gap
        ),
        "causal_recommendation_contrast": (
            comparison.causal_recommendation_contrast
            >= benchmark.minimum_churn_contrast
        ),
        "observational_recommendation_contrast": (
            comparison.observational_recommendation_contrast
            <= -benchmark.minimum_churn_contrast
        ),
        "left_u_shape": (
            comparison.causal_left_contrast >= benchmark.minimum_shoulder_contrast
        ),
        "right_u_shape": (
            comparison.causal_right_contrast >= benchmark.minimum_shoulder_contrast
        ),
        "overlap": all(overlap_passes),
    }
    if intervals is not None:
        gates["causal_interval_excludes_zero"] = intervals.causal.lower > 0.0
        gates["observational_interval_excludes_zero"] = (
            intervals.observational.upper < 0.0
        )
    report: dict[str, object] = {
        "passed": all(gates.values()),
        "gates": gates,
        "grid": comparison.grid.tolist(),
        "causal": comparison.causal.tolist(),
        "observational": comparison.observational.tolist(),
        "causal_optimum": comparison.causal_optimum,
        "observational_optimum": comparison.observational_optimum,
        "recommendation_gap": comparison.recommendation_gap,
        "causal_recommendation_contrast": (
            comparison.causal_recommendation_contrast
        ),
        "observational_recommendation_contrast": (
            comparison.observational_recommendation_contrast
        ),
        "causal_left_contrast": comparison.causal_left_contrast,
        "causal_right_contrast": comparison.causal_right_contrast,
        "overlap": [
            {
                "e": diagnostic.e,
                "effective_sample_size": diagnostic.effective_sample_size,
                "effective_sample_fraction": diagnostic.effective_sample_fraction,
                "maximum_normalized_weight": diagnostic.maximum_normalized_weight,
                "passed": passed,
            }
            for diagnostic, passed in zip(comparison.overlap, overlap_passes)
        ],
    }
    if intervals is not None:
        report["contrast_intervals"] = {
            "causal": {
                "estimate": intervals.causal.estimate,
                "lower": intervals.causal.lower,
                "upper": intervals.causal.upper,
            },
            "observational": {
                "estimate": intervals.observational.estimate,
                "lower": intervals.observational.lower,
                "upper": intervals.observational.upper,
            },
        }
    return report


def evaluate_landmark_benchmark(
    propensity_model: WinPropensityModel,
    *,
    n_players: int,
    seed: int,
    assignment: AssignmentConfig | AssignmentSchedule = ASSIGNMENT_SCHEDULE,
    churn_config: ChurnConfig | ChurnSchedule = CHURN_SCHEDULE,
    mastery_config: MasteryConfig = MasteryConfig(),
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
    n_bootstrap: int = 0,
) -> dict[str, object]:
    """Generate one landmark cohort and evaluate every level's oracle gates."""
    cohort = generate_landmark_cohort(
        propensity_model,
        n_players=n_players,
        seed=seed,
        assignment=assignment,
        churn_config=churn_config,
        mastery_config=mastery_config,
        benchmark=benchmark,
    )
    levels: dict[str, object] = {}
    for risk_set in cohort.risk_sets:
        current_churn = _churn_for_level(churn_config, risk_set.level_name)
        current_assignment = _assignment_for_level(
            assignment, risk_set.level_name
        )
        comparison = compare_oracle_curves(
            risk_set,
            propensity_model,
            churn_config=current_churn,
            mastery_config=mastery_config,
            benchmark=benchmark,
        )
        intervals = (
            bootstrap_recommendation_contrasts(
                risk_set,
                propensity_model,
                comparison,
                churn_config=current_churn,
                mastery_config=mastery_config,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            if n_bootstrap
            else None
        )
        level_report = comparison_report(
            comparison, benchmark, intervals
        )
        level_report["assignment"] = {
            "skill_gain": current_assignment.skill_gain,
            "sigma": current_assignment.sigma,
        }
        level_report["churn"] = {
            "intercept": current_churn.intercept,
            "deviation_coefficient": current_churn.deviation_coefficient,
            "mastery_target": current_churn.mastery_target,
        }
        levels[risk_set.level_name] = level_report
    return {
        "schema_version": 2,
        "passed": all(bool(report["passed"]) for report in levels.values()),
        "seed": seed,
        "n_initial_players": cohort.n_initial_players,
        "n_landmark_players": cohort.n_active_players,
        "landmark_attempt": benchmark.landmark_attempt,
        "warmup_churn_scale": benchmark.warmup_churn_scale,
        "mastery": asdict(mastery_config),
        "benchmark": asdict(benchmark),
        "levels": levels,
    }


def evaluate_validation_suite(
    propensity_model: WinPropensityModel,
    *,
    n_players: int,
    assignment: AssignmentConfig | AssignmentSchedule = ASSIGNMENT_SCHEDULE,
    churn_config: ChurnConfig | ChurnSchedule = CHURN_SCHEDULE,
    mastery_config: MasteryConfig = MasteryConfig(),
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
    n_bootstrap: int = 0,
) -> dict[str, object]:
    """Evaluate every preregistered validation seed without averaging gates."""
    seed_reports = [
        evaluate_landmark_benchmark(
            propensity_model,
            n_players=n_players,
            seed=seed,
            assignment=assignment,
            churn_config=churn_config,
            mastery_config=mastery_config,
            benchmark=benchmark,
            n_bootstrap=n_bootstrap,
        )
        for seed in benchmark.validation_seeds
    ]
    return {
        "schema_version": 2,
        "passed": all(bool(report["passed"]) for report in seed_reports),
        "validation_seeds": list(benchmark.validation_seeds),
        "n_players_per_seed": n_players,
        "n_bootstrap": n_bootstrap,
        "benchmark": asdict(benchmark),
        "seed_reports": seed_reports,
    }


def main() -> None:  # pragma: no cover - CLI
    import argparse

    from .calibrate import WIN_PROPENSITY_PATH, load_win_propensity_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--propensity", type=Path, default=WIN_PROPENSITY_PATH)
    parser.add_argument("--players", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--all-validation-seeds", action="store_true")
    parser.add_argument("--skill-gain", type=float, default=None)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--deviation-coefficient", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()

    model = load_win_propensity_model(args.propensity)
    if args.all_validation_seeds and args.seed is not None:
        parser.error("--seed cannot be combined with --all-validation-seeds")
    if (args.skill_gain is None) != (args.sigma is None):
        parser.error("--skill-gain and --sigma must be supplied together")
    assignment = (
        ASSIGNMENT_SCHEDULE
        if args.skill_gain is None
        else AssignmentConfig(args.skill_gain, args.sigma)
    )
    churn_config = (
        CHURN_SCHEDULE
        if args.deviation_coefficient is None
        else ChurnConfig(deviation_coefficient=args.deviation_coefficient)
    )
    if args.all_validation_seeds:
        report = evaluate_validation_suite(
            model,
            n_players=args.players,
            assignment=assignment,
            churn_config=churn_config,
            n_bootstrap=args.bootstrap,
        )
    else:
        report = evaluate_landmark_benchmark(
            model,
            n_players=args.players,
            seed=args.seed or BENCHMARK_CONFIG.validation_seeds[0],
            assignment=assignment,
            churn_config=churn_config,
            n_bootstrap=args.bootstrap,
        )
    report["propensity_artifact"] = {
        "name": args.propensity.name,
        "sha256": hashlib.sha256(args.propensity.read_bytes()).hexdigest(),
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
        print(f"wrote {args.out}  passed={report['passed']}")
    if args.require_pass and not report["passed"]:
        raise SystemExit(1)


__all__ = [
    "ASSIGNMENT_SCHEDULE",
    "AssignmentConfig",
    "AssignmentSchedule",
    "ContrastInterval",
    "CurveComparison",
    "EngineOutcomeSurface",
    "LandmarkCohort",
    "LandmarkRiskSet",
    "OverlapDiagnostic",
    "RecommendationContrastIntervals",
    "bootstrap_engine_curves",
    "bootstrap_recommendation_contrasts",
    "compare_engine_curves",
    "compare_oracle_curves",
    "comparison_report",
    "evaluate_landmark_benchmark",
    "evaluate_validation_suite",
    "engine_outcome_surface",
    "generate_engine_landmark_cohort",
    "generate_landmark_cohort",
    "oracle_causal_curve",
    "oracle_observational_curve",
    "overlap_diagnostic",
    "passes_initial_gates",
    "randomized_assignment_control",
    "load_engine_outcome_surface",
    "save_engine_outcome_surface",
    "select_grid_optimum",
]


if __name__ == "__main__":  # pragma: no cover
    main()