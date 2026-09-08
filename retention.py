"""Win-propensity and churn mechanisms for the landmark causal query."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyro
import pyro.distributions as dist
import torch

from .spec import BENCHMARK_CONFIG, BenchmarkConfig, SKILL_NAMES, PlayerSkill


def _sigmoid(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-array))


@dataclass(frozen=True)
class ChurnConfig:
    """Symmetric challenge-mismatch hazard configuration."""

    intercept: float = -4.0
    deviation_coefficient: float = 32.0
    target_win_probability: float = 0.55

    def __post_init__(self) -> None:
        if self.deviation_coefficient <= 0:
            raise ValueError("deviation_coefficient must be positive")
        if not 0.0 < self.target_win_probability < 1.0:
            raise ValueError("target_win_probability must lie in (0, 1)")


@dataclass(frozen=True)
class ChurnSchedule:
    """Level-specific mismatch sensitivity with one global win target."""

    level_names: tuple[str, ...] = ("orchard", "harbour", "foundry")
    intercepts: tuple[float, ...] = (-4.0, -4.0, -4.0)
    deviation_coefficients: tuple[float, ...] = (128.0, 64.0, 64.0)
    target_win_probability: float = 0.55

    def __post_init__(self) -> None:
        n_levels = len(self.level_names)
        if len(set(self.level_names)) != n_levels:
            raise ValueError("level_names must be unique")
        if len(self.intercepts) != n_levels:
            raise ValueError("intercepts must align with levels")
        if len(self.deviation_coefficients) != n_levels:
            raise ValueError("deviation coefficients must align with levels")
        if any(value <= 0 for value in self.deviation_coefficients):
            raise ValueError("deviation coefficients must be positive")
        if not 0.0 < self.target_win_probability < 1.0:
            raise ValueError("target_win_probability must lie in (0, 1)")

    def for_level(self, level_name: str) -> ChurnConfig:
        index = self.level_names.index(level_name)
        return ChurnConfig(
            intercept=self.intercepts[index],
            deviation_coefficient=self.deviation_coefficients[index],
            target_win_probability=self.target_win_probability,
        )


CHURN_SCHEDULE = ChurnSchedule()


@dataclass(frozen=True)
class WinPropensityModel:
    """Frozen logistic response surface calibrated against engine rollouts."""

    level_names: tuple[str, ...]
    tier_names: tuple[str, ...]
    intercepts: tuple[tuple[float, ...], ...]
    skill_coefficients: tuple[tuple[float, ...], ...]
    difficulty_coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        n_levels = len(self.level_names)
        n_tiers = len(self.tier_names)
        if len(set(self.level_names)) != n_levels:
            raise ValueError("level_names must be unique")
        if len(set(self.tier_names)) != n_tiers:
            raise ValueError("tier_names must be unique")
        if np.shape(self.intercepts) != (n_levels, n_tiers):
            raise ValueError("intercepts must have shape (levels, tiers)")
        if np.shape(self.skill_coefficients) != (n_levels, len(SKILL_NAMES)):
            raise ValueError("skill_coefficients must have shape (levels, skills)")
        if np.shape(self.difficulty_coefficients) != (n_levels,):
            raise ValueError("difficulty_coefficients must have shape (levels,)")
        if np.any(np.asarray(self.skill_coefficients) < 0):
            raise ValueError("skill coefficients must be non-negative")
        if np.any(np.asarray(self.difficulty_coefficients) <= 0):
            raise ValueError("difficulty coefficients must be positive")

    def logits(
        self,
        level_name: str,
        tier_indices: np.ndarray,
        skills: np.ndarray,
        served_difficulty: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate response-surface logits for one level."""
        level_index = self.level_names.index(level_name)
        tiers = np.asarray(tier_indices, dtype=np.int64)
        skill_array = np.asarray(skills, dtype=np.float64)
        if skill_array.ndim != 2 or skill_array.shape[1] != len(SKILL_NAMES):
            raise ValueError("skills must have shape (players, skills)")
        if tiers.shape != (skill_array.shape[0],):
            raise ValueError("tier_indices must align with skills")
        if np.any((tiers < 0) | (tiers >= len(self.tier_names))):
            raise ValueError("tier index outside model support")
        served = np.asarray(served_difficulty, dtype=np.float64)
        if served.ndim > 1 or (served.ndim == 1 and served.shape != tiers.shape):
            raise ValueError("served_difficulty must be scalar or align with skills")

        intercept = np.asarray(self.intercepts[level_index])[tiers]
        coefficient = np.asarray(self.skill_coefficients[level_index])
        return (
            intercept
            + skill_array @ coefficient
            - self.difficulty_coefficients[level_index] * served
        )

    def probabilities(
        self,
        level_name: str,
        tier_indices: np.ndarray,
        skills: np.ndarray,
        served_difficulty: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate oracle win propensity for one level."""
        return _sigmoid(
            self.logits(level_name, tier_indices, skills, served_difficulty)
        )

    def probability(
        self,
        level_name: str,
        tier_index: int,
        skill: PlayerSkill,
        served_difficulty: float,
    ) -> float:
        values = self.probabilities(
            level_name,
            np.asarray([tier_index]),
            skill.as_array()[None, :],
            served_difficulty,
        )
        return float(values[0])

    def to_dict(self) -> dict[str, object]:
        return {
            "level_names": list(self.level_names),
            "tier_names": list(self.tier_names),
            "intercepts": [list(row) for row in self.intercepts],
            "skill_coefficients": [list(row) for row in self.skill_coefficients],
            "difficulty_coefficients": list(self.difficulty_coefficients),
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "WinPropensityModel":
        return cls(
            level_names=tuple(str(value) for value in values["level_names"]),
            tier_names=tuple(str(value) for value in values["tier_names"]),
            intercepts=tuple(
                tuple(float(value) for value in row)
                for row in values["intercepts"]
            ),
            skill_coefficients=tuple(
                tuple(float(value) for value in row)
                for row in values["skill_coefficients"]
            ),
            difficulty_coefficients=tuple(
                float(value) for value in values["difficulty_coefficients"]
            ),
        )


@dataclass(frozen=True)
class AttemptRecord:
    """One ordered episode plus its post-episode churn decision."""

    player_id: int
    attempt_id: int
    episode: object
    win_probability: float
    churn_probability: float
    churn_after: int


@dataclass(frozen=True)
class PlayerTrajectory:
    """Ordered attempts sharing one player-level skill draw."""

    player_id: int
    player: PlayerSkill
    attempts: tuple[AttemptRecord, ...]

    @property
    def churned(self) -> bool:
        return bool(self.attempts and self.attempts[-1].churn_after)

    @property
    def churn_attempt(self) -> int | None:
        return self.attempts[-1].attempt_id if self.churned else None


def challenge_mismatch_hazard(
    win_probability: np.ndarray | float,
    config: ChurnConfig = ChurnConfig(),
) -> np.ndarray:
    """Return churn probability, minimized at the configured win target."""
    probability = np.asarray(win_probability, dtype=np.float64)
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("win probability must lie in [0, 1]")
    logit = config.intercept + config.deviation_coefficient * (
        probability - config.target_win_probability
    ) ** 2
    return _sigmoid(logit)


def sample_C(
    win_probability: float,
    *,
    active: bool = True,
    config: ChurnConfig = ChurnConfig(),
    name: str = "C",
    value: int | None = None,
) -> int:
    """Sample absorbing churn status after an attempt."""
    if not active:
        return int(pyro.deterministic(name, torch.tensor(1)))
    hazard = float(challenge_mismatch_hazard(win_probability, config))
    if value is not None:
        outcome = pyro.deterministic(name, torch.tensor(int(value)))
    else:
        outcome = pyro.sample(name, dist.Bernoulli(torch.tensor(hazard)))
    return int(outcome)


def simulate_player_trajectory(
    propensity_model: WinPropensityModel,
    *,
    player_id: int,
    seed: int,
    max_attempts: int = 30,
    benchmark: BenchmarkConfig = BENCHMARK_CONFIG,
    churn_config: ChurnConfig | ChurnSchedule = CHURN_SCHEDULE,
    player: PlayerSkill | None = None,
) -> PlayerTrajectory:
    """Simulate ordered attempts with the churn clock starting at the landmark."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    from .scm import TIER_NAMES, ground_truth_model, sample_K

    skill_seed = int(
        np.random.SeedSequence([seed, player_id, 0]).generate_state(1)[0]
    )
    pyro.set_rng_seed(skill_seed)
    skill = sample_K(value=player)
    attempts: list[AttemptRecord] = []
    for attempt_id in range(1, max_attempts + 1):
        episode_seed = int(
            np.random.SeedSequence([seed, player_id, attempt_id]).generate_state(1)[0]
        )
        pyro.set_rng_seed(episode_seed)
        episode = ground_truth_model(player=skill)
        tier_index = TIER_NAMES.index(episode.tier)
        win_probability = propensity_model.probability(
            episode.level.name,
            tier_index,
            skill,
            episode.E,
        )
        if attempt_id < benchmark.landmark_attempt:
            churn_probability = 0.0
            churn_after = 0
        else:
            current_churn_config = (
                churn_config.for_level(episode.level.name)
                if isinstance(churn_config, ChurnSchedule)
                else churn_config
            )
            churn_probability = float(
                challenge_mismatch_hazard(win_probability, current_churn_config)
            )
            churn_after = sample_C(
                win_probability,
                config=current_churn_config,
                name=f"C/{attempt_id}",
            )
        attempts.append(
            AttemptRecord(
                player_id=player_id,
                attempt_id=attempt_id,
                episode=episode,
                win_probability=win_probability,
                churn_probability=churn_probability,
                churn_after=churn_after,
            )
        )
        if churn_after:
            break
    return PlayerTrajectory(player_id, skill, tuple(attempts))


__all__ = [
    "AttemptRecord",
    "CHURN_SCHEDULE",
    "ChurnConfig",
    "ChurnSchedule",
    "PlayerTrajectory",
    "WinPropensityModel",
    "challenge_mismatch_hazard",
    "sample_C",
    "simulate_player_trajectory",
]