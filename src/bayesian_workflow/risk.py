import numpy as np


def posterior_predictive_losses(p_silent_draws: np.ndarray, observed_errors: np.ndarray, conversion: float = 0.05, draws: int = 20_000, seed: int = 20260923) -> np.ndarray:
    rng = np.random.default_rng(seed)
    p = rng.choice(np.asarray(p_silent_draws, float), size=draws, replace=True)
    errors = np.abs(np.asarray(observed_errors, float)); errors = errors[np.isfinite(errors) & (errors > 0)]
    severity = rng.choice(errors, size=draws, replace=True) * conversion if len(errors) else np.ones(draws)
    return (rng.random(draws) < p) * severity


def var_es(losses: np.ndarray, level: float = 0.995) -> tuple[float, float]:
    var = float(np.quantile(losses, level)); tail = losses[losses >= var]
    return var, float(tail.mean()) if len(tail) else var
