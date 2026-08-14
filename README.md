# Causal Match-3 Simulator

A match-3 simulator specified as an explicit causal generative model. The model
is the data-generating process, providing known ground truth for causal inference
and world-model experiments.

## Model

- `L`: level context
- `D`: baseline difficulty tier
- `K`: latent player-skill segment
- `E`: difficulty served by the DDA policy
- `S_t`: board state, goal colour, moves remaining, and goals remaining
- `A_t`: adjacent-tile swap
- `X`: behavioural proxies for `K`
- `R`: indicator that the player completed the level within the move budget

The simulator targets `P(R = 1 | do(E = e), L = l)`. In the observational data,
the DDA serves harder levels to stronger players, while skill also improves move
selection. Consequently, `P(R | E)` differs from `P(R | do(E))`.

## Install

```bash
pip install -e .
```

## Use

```python
import pyro
import match3_simulator as match3

pyro.set_rng_seed(0)
episode = match3.ground_truth_model(
    level=match3.LEVELS[0],
    player=match3.SEGMENTS[2],
    E=0.0,
)
print(episode.R, episode.moves_used, episode.served_goal_count)
```

Generate a dataset:

```bash
match3-simulate -n 2000 --out data
```

Render an episode:

```bash
match3-video --seed 12 --level orchard --segment regular \
  --E 0 --out episode.mp4 --json episode.json --thumb opening.png
```

Recalibrate the mapping from served difficulty to goal count:

```bash
python -m match3_simulator.calibrate --n 150
```

## Causal controls

`ground_truth_model(dda_gain=...)` controls how strongly the assignment policy
adapts difficulty to skill. `ground_truth_model(e_sigma=...)` controls residual
within-stratum variation in served difficulty. Passing `E=e` clamps the treatment
and samples from `do(E=e)`.
