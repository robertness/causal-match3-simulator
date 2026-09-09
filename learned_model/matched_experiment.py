"""Small matched-arm smoke experiment for generative model integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import torch

from ..calibrate import load_win_propensity_model
from ..retention import simulate_player_trajectory
from ..spec import BENCHMARK_CONFIG
from .action_policy import ActionPolicyConfig
from .arms import GenerativeModelConfig, GenerativeWorldModel, ModelArm
from .batching import build_generative_training_batch
from .encoder import PrefixEncoderConfig
from .generative import GameplayRSSMConfig
from .train import train_generative_world_model_step


@dataclass(frozen=True)
class MatchedGenerativeSmokeConfig:
    n_players: int = 3
    target_attempt: int = 2
    embedding_size: int = 8
    observation_size: int = 16
    task_context_size: int = 8
    hidden_size: int = 16
    stochastic_size: int = 8
    behavior_size: int = 16
    prefix_hidden_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    context_kl_weight: float = 0.2
    dynamics_kl_weight: float = 0.2
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        integer_fields = (
            "n_players",
            "target_attempt",
            "embedding_size",
            "observation_size",
            "task_context_size",
            "hidden_size",
            "stochastic_size",
            "behavior_size",
            "prefix_hidden_size",
        )
        if any(getattr(self, name) < 1 for name in integer_fields):
            raise ValueError("smoke dimensions and counts must be positive")
        if self.behavior_size % 2:
            raise ValueError("behavior_size must be divisible by two heads")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning rate and gradient clip must be positive")


def _model_config(config: MatchedGenerativeSmokeConfig) -> GenerativeModelConfig:
    return GenerativeModelConfig(
        dynamics=GameplayRSSMConfig(
            embedding_size=config.embedding_size,
            observation_size=config.observation_size,
            task_context_size=config.task_context_size,
            hidden_size=config.hidden_size,
            stochastic_size=config.stochastic_size,
        ),
        prefix=PrefixEncoderConfig(hidden_size=config.prefix_hidden_size),
        behavior=ActionPolicyConfig(
            d_model=config.behavior_size,
            n_layers=1,
            n_heads=2,
        ),
    )


def run_matched_generative_smoke(
    config: MatchedGenerativeSmokeConfig,
    *,
    output_dir: str | Path,
) -> dict[str, object]:
    """Simulate one shared batch and update every generative arm once."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    propensity_model = load_win_propensity_model()
    benchmark = replace(BENCHMARK_CONFIG, warmup_churn_scale=0.0)
    trajectories = [
        simulate_player_trajectory(
            propensity_model,
            player_id=player_id,
            seed=config.seed,
            max_attempts=config.target_attempt,
            benchmark=benchmark,
        )
        for player_id in range(config.n_players)
    ]
    prefix, target, transitions, oracle_skill = build_generative_training_batch(
        trajectories,
        target_attempt=config.target_attempt,
        device=device,
    )
    model_config = _model_config(config)
    arm_reports: dict[str, object] = {}
    for arm in ModelArm:
        torch.manual_seed(config.seed)
        model = GenerativeWorldModel(arm, model_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        started = time.perf_counter()
        metrics = train_generative_world_model_step(
            model,
            prefix=prefix,
            target=target,
            transitions=transitions,
            optimizer=optimizer,
            oracle_skill=oracle_skill if arm is ModelArm.ORACLE else None,
            context_kl_weight=config.context_kl_weight,
            dynamics_kl_weight=config.dynamics_kl_weight,
            gradient_clip=config.gradient_clip,
        )
        runtime = time.perf_counter() - started
        checkpoint_name = f"{arm.value}.pt"
        torch.save(
            {
                "schema_version": 1,
                "arm": arm.value,
                "config": asdict(model_config),
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
            },
            output / checkpoint_name,
        )
        arm_reports[arm.value] = {
            **metrics,
            "parameters": model.parameter_counts(),
            "runtime_seconds": runtime,
            "checkpoint": checkpoint_name,
            "seed": config.seed,
        }
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "smoke",
        "config": asdict(config),
        "n_players": len(trajectories),
        "target_attempt": config.target_attempt,
        "n_target_transitions": int(transitions["step_mask"].sum()),
        "arms": arm_reports,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


__all__ = ["MatchedGenerativeSmokeConfig", "run_matched_generative_smoke"]
