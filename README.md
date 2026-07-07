# EstimationBayesian

This project has two supported run modes.

## 1. Run Batch

Edit jobs in `BatchSettings.py`, then run:

```bash
python3 run_batch.py
```

Batch defaults live in `BatchSettings.DEFAULTS`; each entry in `RUNS` can
override them. Common fields:

- `method`: `tk` or `prelec`
- `utility`: `cara` or `power`
- `discounting`: `hyperbolic`, `exponential`, or `quasi_hyperbolic`
- `clusters`, `draws`, `chains`, `cores`, `seed`
- `ksi_mode`: `marginalized`, `fixed`, or `estimated`
- `delta_bounds`: use equal endpoints to fix structural delta, or an interval to estimate it

Each completed run writes only summaries, diagnostics, plots, logs, and settings.
New runs do not save `trace.nc`.

Main outputs per run:

- `settings.json`
- `status.json`
- `run.log`
- `summary.csv`
- `summary_reference_weights.csv`
- `summary_model_fit.csv`
- `summary_diagnostics.csv`
- `plots/`

## 2. Run Single

Run one estimation for each fixed reference point, then compare SQ, PA, LE, and FE:

```bash
python3 run_single_reference_points.py
```

`run_single_reference_points.py` inherits its default model settings from
`BatchSettings.DEFAULTS`. Command-line options override those defaults:

```bash
python3 run_single_reference_points.py --utility power --delta-bounds 1e-6,1.0 --rerun
```

Main combined outputs:

- `single_reference_comparison.csv`
- `single_reference_cluster_params.csv`
- `single_reference_fixed_weights.csv`
- `single_reference_model_fit.csv`

Each fixed-reference subfolder also contains the same per-run summaries,
diagnostics, plots, logs, and settings as a batch run. New runs do not save
`trace.nc`.
