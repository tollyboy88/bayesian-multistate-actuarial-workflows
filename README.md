# Bayesian Multistate Actuarial Workflows

This repository contains the statistical methodology, reproducible analyses and interactive visualisation for the paper **Bayesian Multistate Modelling of Error Propagation and Control Placement in Automated Actuarial Workflows**.

It is deliberately separate from the applied fault-injection testbed used by Paper 1. Paper 2 estimates workflow-state transition distributions, models the magnitude of persistent errors, validates interval recovery and held-out prediction, and solves a finite control-placement problem. Both papers use the same open dataset archive: [Zenodo DOI 10.5281/zenodo.22821051](https://doi.org/10.5281/zenodo.22821051).

## Main results

- 6,144 stage-level transition observations from eight held-out-capable simulation worlds
- hierarchical held-out log score 0.338 versus 1.076 for the global baseline
- expanded 27-cell, four-parameter recovery coverage of 86.0% and 91.7% at nominal 90% and 95%, documenting modest undercoverage
- analytic recursion within 0.000103 of Monte Carlo simulation
- budget-feasible control placement `010101`, with mean terminal active-error probability 0.209, 99.3% posterior constraint probability and 9.0% posterior probability of optimality
- stable ES₀.₉₉₅ ranking when predictive simulations increase from 1,000 to 10,000 per posterior draw (all-placement Spearman correlation 0.992)

These are controlled-study results, not estimates of production insurer failure frequencies.

## Install and reproduce

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test,dashboard]"
.venv\Scripts\python scripts\reproduce.py
.venv\Scripts\python -m pytest
```

The frozen transition table in `data/transition_observations.parquet` is derived from the shared Zenodo archive. `scripts/reproduce.py` regenerates the posterior summaries, validation tables, optimisation results, figures and checksum manifest.

## Bayesian Workflow Laboratory

Paper 2 has a separate light academic dashboard focused on posterior uncertainty and decision analysis:

```powershell
.venv\Scripts\streamlit run dashboard\app.py
```

Open `http://localhost:8501`. The dashboard is intentionally different from Paper 1's operational control-room interface.

## Repository boundary

- **This repository:** multistate Bayesian inference, Markov-additive error magnitude, calibration, prediction and control placement.
- **Paper 1 repository:** experimental workflow engine, fault injection, replay controls, operational-risk scenarios and the operational dashboard.
- **Shared resource:** the same open Zenodo data archive, cited by DOI rather than duplicated as a second research dataset.

## Licence

MIT. See `LICENSE`.
