# Legacy churn artifacts

Files in this directory predate the experienced-mastery data-generating
process. They model churn from an oracle win-propensity value, use an
observation-only warm-up, and must not be cited as results for the current
`R -> M -> C` benchmark.

`landmark-validation-report.json` is preserved unchanged for provenance. Its
`schema_version: 1`, `warmup_churn_scale: 0.0`, and
`target_win_probability: 0.55` fields identify the superseded mechanism.