from __future__ import annotations
import numpy as np

def posterior_predictive_losses(p_silent_draws: np.ndarray, observed_errors: np.ndarray,
    log_increment_draws: np.ndarray | None = None, conversion: float = 0.05,
    exposure: float = 1_000_000.0, samples_per_draw: int = 1_000, seed: int = 20260923) -> np.ndarray:
    """Draw conditional annual losses while retaining parameter uncertainty by row."""
    rng = np.random.default_rng(seed)
    p = np.clip(np.asarray(p_silent_draws, float), 0.0, 1.0)
    errors = np.abs(np.asarray(observed_errors, float))
    errors = errors[np.isfinite(errors) & (errors > 0)]
    if not len(errors): errors = np.array([0.01])
    increments = np.zeros_like(p) if log_increment_draws is None else np.asarray(log_increment_draws, float)
    base = rng.choice(errors, size=(len(p), samples_per_draw), replace=True)
    active = rng.random((len(p), samples_per_draw)) < p[:, None]
    return active * base * np.exp(np.clip(increments[:, None], -8.0, 8.0)) * conversion * exposure

def var_es(losses: np.ndarray, level: float = 0.995) -> tuple[np.ndarray, np.ndarray]:
    values = np.atleast_2d(np.asarray(losses, float))
    var = np.quantile(values, level, axis=1)
    es = np.array([row[row >= threshold].mean() for row, threshold in zip(values, var)])
    return var, es
