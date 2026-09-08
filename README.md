# Causal Match-3 Simulator

A match-3 simulator specified as an explicit causal generative model. The model
is the data-generating process, providing known ground truth for causal inference
and world-model experiments.

## Model

- `L`: level context
- `D`: baseline difficulty tier
- `K`: stable continuous player skill: search, pattern, planning, and strategy
- `E`: difficulty served by the DDA policy
- `S_t`: board state, goal colour, moves remaining, and goals remaining
- `A_t`: adjacent-tile swap
- `X`: typed task evidence generated through a sparse multidimensional Q-matrix
- `R`: indicator that the player completed the level within the move budget
- `C`: absorbing churn after the landmark attempt

The primary target is the level-specific difficulty minimizing next-attempt churn:

```text
argmin_e P(C_i,21 = 1 | do(E_i,20 = e), L_i,20 = l)
```

Attempts 1--19 form an observation-only warm-up in the primary benchmark. Churn
is U-shaped in oracle win propensity around a target of `0.55`: play that is too
easy or too hard raises churn. In natural data, the DDA serves harder content to
stronger players, while skill also improves action quality. The observational
and interventional churn-optimal settings are therefore deliberately different.

`D` is sampled from `p(D | L)`: there is no `K -> D` edge. Skill adaptation acts
through `K -> E`, and the tier has a separate state effect through its move
budget.

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

Fit the engine-derived oracle win-propensity surface:

```bash
python -m match3_simulator.calibrate --win-propensity \
  --skill-draws 256 --replicates 2 --workers 6
```

Generate ordered player histories after that artifact exists:

```bash
match3-simulate --mode players -n 100 --max-attempts 30 --out data/players
```

Validate every preregistered landmark seed and render the representative curve:

```bash
python -m match3_simulator.causal_queries --all-validation-seeds \
  --players 20000 --bootstrap 200 --require-pass \
  --out data/landmark-validation-report.json
python -m match3_simulator.plot_queries \
  data/landmark-validation-report.json --out media/landmark_churn_curves.png
```

The `learned_model` package contains the continuous strict-prefix encoder,
support-aware structural decoder heads, exact legal-action masking, spatial
action transformer, and CPU-capable training loops. Its deployable path never
receives true `K` or oracle win propensity.

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
within-stratum variation in served difficulty. Passing `E=e` clamps the treatment
and samples from `do(E=e)`.

The validated natural assignment uses level-specific `(gain, residual sd)`:
`orchard=(1.5, 0.65)`, `harbour=(0.8, 0.70)`, and
`foundry=(1.0, 1.00)`. The churn target remains globally `0.55`, while mismatch
curvature is `(128, 64, 64)` for the same level order. These schedules pass the
recommendation-reversal, bilateral U-shape, bootstrap-direction, and overlap
gates on all preregistered validation seeds for the current engine response
surface.
