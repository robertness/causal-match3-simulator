# Causal Match-3 Simulator

A match-3 simulator specified as an explicit causal generative model. The model
is the data-generating process, providing known ground truth for causal inference
and world-model experiments.

## Model

- `L`: level context
- `D`: baseline difficulty tier
- `K`: stable continuous player skill: search, pattern, planning, and strategy
- `E`: difficulty served by the DDA policy
- `S_t`: color and special-kind grids, goal colour, moves remaining, and goals remaining
- `A_t`: adjacent-tile swap
- `X`: typed task evidence generated through a sparse multidimensional Q-matrix
- `R`: indicator that the player completed the level within the move budget
- `M`: recency-weighted expected-experience state updated from realized completion
- `Q`: signed completion margin for early wins and unmet quotas
- `C`: absorbing next-attempt churn conditioned on post-attempt `M` and `Q`

The primary target is the level-specific difficulty minimizing next-attempt churn:

```text
argmin_e P(C_i,21 = 1 | do(E_i,20 = e), L_i,20 = l)
```

Attempts 1--19 form the strict history prefix for the landmark query. After each
completed attempt, realized completion updates expected experience,
`M_next = M + rho * (R - M)`, and active players face a level-specific,
two-sided churn hazard whose over-challenge curvature may exceed its easy-side
curvature. The frozen win-propensity model
is retained only as an engine-response diagnostic; it is not a parent of churn.
In natural data, the DDA serves harder content to stronger players, while skill
also improves action quality. Calibration pilots that fail the preregistered
engine gates remain pilot artifacts rather than benchmark results.

`D` is sampled from `p(D | L)`: there is no `K -> D` edge. Skill adaptation acts
through `K -> E`, and the tier has a separate state effect through its move
budget.

Runs of four created by a swap produce horizontal or vertical striped tiles.
When matched, stripes clear their row or column and can trigger other stripes
in the same cascade round. Color and special kind are stored separately so
specials do not alter color-match detection.

## Install

```bash
pip install -e '.[test]'
```

## Use

```python
import pyro
import match3_simulator as match3

pyro.set_rng_seed(0)
episode = match3.ground_truth_model(
    level=match3.LEVELS[0],
    player=match3.PlayerSkill((0.2, 0.8, -0.1, 0.5), "example"),
    E=0.0,
)
print(episode.player.as_dict(), episode.R, episode.moves_used)
```

Generate a dataset:

```bash
match3-simulate -n 2000 --out data
```

Render an episode:

```bash
match3-video --seed 12 --level orchard --profile planner \
  --E 0 --out episode.mp4 --json episode.json --thumb opening.png
```

Recalibrate and independently validate the mapping from served difficulty to
goal count:

```bash
python -m match3_simulator.calibrate --n 1000 --workers 6
python -m match3_simulator.calibrate --validate-goals --n 1000 --workers 6 \
  --out /tmp/goal-calibration-validation.json
```

Calibration variants can be selected without replacing the checked-in table:

```bash
MATCH3_CALIBRATION_PATH=/path/to/calibration.json \
  python -m match3_simulator.engine_benchmark --warmup-only ...
```

The selected file may be a direct level table or a manifest containing a
`table` field. Engine reports record its resolved path and SHA-256, and direct
benchmark configuration hashes include that provenance.

Fit the engine-derived oracle win-propensity surface:

```bash
python -m match3_simulator.calibrate --win-propensity \
  --skill-draws 256 --replicates 2 --workers 6
```

Generate ordered player histories after that artifact exists:

```bash
match3-simulate --mode players -n 100 --max-attempts 30 --out data/players
```

This writes deployable `episodes.csv` and `transitions.npz`, a separate
simulator-only `oracle/attempts.csv`, and a manifest containing configuration,
row counts, code revision, and artifact hashes. True `K`, true `M`, oracle win
propensity, and churn probabilities never appear in either deployable artifact.
The transition artifact includes level, tier, served difficulty, current and
next special grids, next-state counters, and episode/player identifiers so `load_gameplay_transition_dataset`
can construct masked `GameplayRSSM` sequence batches directly.

Run a board-engine calibration pilot and render its curves:

```bash
python -m match3_simulator.engine_benchmark \
  --players 512 --engine-players 512 --rollouts-per-player 8 \
  --workers 8 --bootstrap 2000 --status pilot \
  --out data/engine-calibration
python -m match3_simulator.plot_queries \
  data/engine-calibration/report.json --out media/landmark_churn_curves.png
```

The engine benchmark freezes `(K,D,L,M)` at the landmark and reuses one
exogenous seed across every candidate `E`. Since `E` changes only the quota, one
full-budget goal total is thresholded over the grid; tests require this optimized
construction to match direct per-`E` engine runs exactly. Whole-player bootstrap
replicates preserve all paired interventions and gameplay replicates.

The `learned_model` package contains the continuous strict-prefix encoder,
support-aware structural decoder heads, exact legal-action masking, spatial
action transformer, and CPU-capable training loops. Its deployable path never
receives true `K` or oracle win propensity.

Its generative package exposes four explicit wrappers around one fast RSSM:
pooled, parameter-matched no-`K`, causal strict-prefix context, and oracle-`K`.
The no-`K`, causal, and oracle wrappers have identical parameter counts. Stable
player context enters the shared behavior policy, while board and counter
mechanics remain conditioned only on observed state, action, level, tier, and
served difficulty.

Run a player-disjoint matched experiment and reload its best checkpoints for
held-out response curves:

```python
from match3_simulator.learned_model import (
  MatchedGenerativeTrainConfig,
  evaluate_matched_experiment_directory,
  run_matched_generative_experiment,
)

run_matched_generative_experiment(
  MatchedGenerativeTrainConfig(n_players=2400, seed=4201),
  output_dir="data/matched-seed4201",
  progress=print,
)
evaluate_matched_experiment_directory(
  "data/matched-seed4201",
  target_attempt=20,
  rollouts_per_player=16,
  seed=4301,
)
```

Training uses one deterministic player split and shared batch order for all
arms, selects a separate best validation checkpoint per arm, and evaluates the
untouched test split. `response-curves.json` keeps structural head
g-computation separate from full learned imagination. The latter samples a
learned task-conditioned opening state, chooses legal actions with the learned
policy, conditions the first transition on that opening, and then advances only
through RSSM priors. Player context is inferred once and held fixed across the
served-difficulty sweep; simulator skill is supplied only to the oracle arm.

Run a player-disjoint CPU smoke experiment and its real-engine closed-loop check:

```bash
python -m match3_simulator.learned_model.experiment \
  --players 32 --max-attempts 20 --target-attempt 20 \
  --action-epochs 5 --vae-epochs 5 --device cpu \
  --out data/learned-smoke
python -m match3_simulator.learned_model.evaluate \
  data/learned-smoke --e-grid -1 0 1
```

The runner fits state-only and oracle-skill action references before the
deployable continuous VAE. Checkpoints and the separate oracle evaluation arrays
remain local; only compact JSON reports are versioned. Teacher-forced gains do
not establish a causal result unless the learned win head also agrees with
closed-loop learned-policy rollouts through the real board engine.

The current 256-player development artifact reports attempt-20 action NLL
`2.237` (state-only), `2.124` (inferred skill), and `2.053` (oracle skill).
Learned causal recommendations have held-out oracle regret `0.000`, `0.004`, and
`0.017` for orchard, harbour, and foundry. The learned win head remains less
calibrated than direct learned-policy engine rollouts, so these are development
results rather than a final causal release.

## Causal controls

`ground_truth_model(dda_gain=...)` controls how strongly the assignment policy
adapts difficulty to skill. `ground_truth_model(e_sigma=...)` controls residual
within-stratum variation in served difficulty. Level-aligned `dda_gains=` and
`e_sigmas=` support calibration runs without changing module defaults. Passing
`E=e` clamps the treatment and samples from `do(E=e)`.

The engine CLI likewise accepts explicit expected-experience, easy/hard-side
churn, assignment, and grid parameters. Parameters selected on calibration seeds are not promoted to live
defaults until all levels pass on every held-out validation seed, including
player-bootstrap direction and overlap gates.
