"""Public API for the match-3 generative simulator."""

import os as _os

# Set before any submodule pulls in pygame, whichever import lands first.
_os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
_os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from .board import (
    apply_swap,
    deal,
    has_match,
    immediate_effect,
    legal_moves,
    match_mask,
    resolve_move,
    settle,
)
from .policy import action_probs, beta_from_skill, move_scores, policy_distribution
from .scm import (
    DDA_GAIN,
    LEVELS,
    PROXY_NAMES,
    TIER_LOGITS,
    TIER_NAMES,
    Episode,
    ground_truth_model,
    sample_A,
    sample_D,
    sample_E,
    sample_K,
    sample_L,
    sample_R,
    sample_S0,
    sample_S_next,
    sample_X,
)
from .spec import (
    SEGMENTS,
    Action,
    CascadeStep,
    Difficulty,
    LevelContext,
    PlayerType,
    State,
    Transition,
)

__version__ = "0.1.0"

__all__ = [
    "Action",
    "CascadeStep",
    "DDA_GAIN",
    "Difficulty",
    "Episode",
    "LEVELS",
    "LevelContext",
    "PROXY_NAMES",
    "PlayerType",
    "SEGMENTS",
    "State",
    "TIER_LOGITS",
    "TIER_NAMES",
    "Transition",
    "action_probs",
    "apply_swap",
    "beta_from_skill",
    "deal",
    "ground_truth_model",
    "has_match",
    "immediate_effect",
    "legal_moves",
    "match_mask",
    "move_scores",
    "policy_distribution",
    "resolve_move",
    "sample_A",
    "sample_D",
    "sample_E",
    "sample_K",
    "sample_L",
    "sample_R",
    "sample_S0",
    "sample_S_next",
    "sample_X",
    "settle",
]
