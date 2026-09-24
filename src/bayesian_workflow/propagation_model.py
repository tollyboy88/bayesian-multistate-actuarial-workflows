from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def fit_markov_additive(observations: pd.DataFrame) -> pd.DataFrame:
    """Fit conjugate Student-t posteriors for active-error log increments."""
    active = observations.loc[
        (observations["state_before"] == "E")
        & (observations["state_after"] == "E")
        & observations["paired_log_increment"].notna()
        & np.isfinite(observations["paired_log_increment"])
    ].copy()
    values = active["paired_log_increment"].to_numpy(float)
    mu0 = float(np.mean(values)) if len(values) else 0.0
    kappa0, alpha0 = 2.0, 2.0
    beta0 = float(np.var(values) + 0.25) if len(values) else 1.0
    rows: list[dict[str, object]] = []
    for key, group in active.groupby(["stage", "fault_type", "topology"]):
        x = group["paired_log_increment"].to_numpy(float)
        n = len(x); mean = float(x.mean()); ss = float(np.square(x - mean).sum())
        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0 + n * mean) / kappa_n
        alpha_n = alpha0 + n / 2
        beta_n = beta0 + 0.5 * ss + (kappa0 * n * (mean - mu0) ** 2) / (2 * kappa_n)
        scale = np.sqrt(beta_n / (alpha_n * kappa_n))
        lo, hi = stats.t.ppf([0.05, 0.95], df=2 * alpha_n, loc=mu_n, scale=scale)
        rows.append({
            "stage": key[0], "fault_type": key[1], "topology": key[2], "n": n,
            "posterior_mean_increment": mu_n, "ci_05": lo, "ci_95": hi,
            "predictive_scale": np.sqrt(beta_n * (kappa_n + 1) / (alpha_n * kappa_n)),
            "degrees_freedom": 2 * alpha_n,
        })
    return pd.DataFrame(rows)
