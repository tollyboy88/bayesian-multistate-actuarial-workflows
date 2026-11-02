from __future__ import annotations
from itertools import product
import numpy as np
import pandas as pd
from .propagation_model import fit_markov_additive
from .risk import posterior_predictive_losses, var_es
from .transition_model import HierarchicalDirichletTransitionModel, STATES

def _severity_draws(summary, stage, topology, faults, rng):
    subset = summary.loc[(summary.stage == stage) & (summary.topology == topology)].set_index("fault_type")
    fallback = summary.loc[summary.topology == topology]
    if fallback.empty: fallback = summary
    mu0 = float(fallback.posterior_mean_increment.mean()) if len(fallback) else 0.0
    scale0 = float(fallback.predictive_scale.median()) if len(fallback) else 1.0
    df0 = float(fallback.degrees_freedom.median()) if len(fallback) else 4.0
    out = np.empty(len(faults))
    for i, fault in enumerate(faults):
        row = subset.loc[fault] if fault in subset.index else None
        if isinstance(row, pd.DataFrame): row = row.iloc[0]
        mu, scale, df = (mu0, scale0, df0) if row is None else (row.posterior_mean_increment, row.predictive_scale, row.degrees_freedom)
        out[i] = mu + scale * rng.standard_t(max(float(df), 2.1))
    return out

def enumerate_stage_controls(observations: pd.DataFrame, stage_cost: float = 400.0, budget: float = 1_200.0,
    risk_limit: float = 0.35, required_probability: float = 0.90, posterior_draws: int = 300,
    samples_per_draw: int = 1_000, seed: int = 20260923,
    model: HierarchicalDirichletTransitionModel | None = None) -> pd.DataFrame:
    """Enumerate 64 placements using posterior ES and a posterior reliability constraint."""
    stages = sorted(observations.stage.unique(), key=lambda s: int(str(s)[1]))
    model = model or HierarchicalDirichletTransitionModel().fit(observations)
    severity = fit_markov_additive(observations)
    active = observations.loc[observations.state_before == "E"]
    weights_s = active.fault_type.value_counts(normalize=True)
    faults, weights = weights_s.index.to_numpy(), weights_s.to_numpy(float)
    observed_errors = observations.get("paired_relative_delta", pd.Series(dtype=float)).to_numpy(float)
    rng = np.random.default_rng(seed); rows = []; es_matrix = []
    for placement_index, placement in enumerate(product([0, 1], repeat=len(stages))):
        persistence = np.ones(posterior_draws); increments = np.zeros(posterior_draws)
        for stage, controlled in zip(stages, placement):
            topology = "T2_VALIDATOR" if controlled else "T1_LINEAR"
            stage_p = np.zeros(posterior_draws)
            for fault, weight in zip(faults, weights):
                stage_p += weight * model.draw_many(stage, str(fault), topology, "E", posterior_draws, rng)[:, STATES.index("E")]
            persistence *= np.clip(stage_p, 1e-8, 1.0)
            increments += _severity_draws(severity, stage, topology, rng.choice(faults, posterior_draws, p=weights), rng)
        losses = posterior_predictive_losses(persistence, observed_errors, increments,
            samples_per_draw=samples_per_draw, seed=seed + placement_index + 1)
        _, es = var_es(losses, 0.995); es_matrix.append(es)
        cost = stage_cost * sum(placement); constraint = float(np.mean(persistence <= risk_limit))
        rows.append({"placement": "".join(map(str, placement)), "controls": sum(placement), "annual_cost": cost,
            "posterior_mean_silent_probability": float(persistence.mean()),
            "silent_probability_ci_05": float(np.quantile(persistence, .05)), "silent_probability_ci_95": float(np.quantile(persistence, .95)),
            "posterior_mean_es_995": float(es.mean()), "es_995_ci_05": float(np.quantile(es, .05)), "es_995_ci_95": float(np.quantile(es, .95)),
            "posterior_probability_risk_constraint": constraint, "feasible": bool(cost <= budget and constraint >= required_probability)})
    result = pd.DataFrame(rows); es_array = np.vstack(es_matrix); feasible = result.feasible.to_numpy()
    if not feasible.any(): feasible = result.annual_cost.to_numpy() <= budget
    objectives = es_array + result.annual_cost.to_numpy()[:, None]; objectives[~feasible] = np.inf
    winners = np.argmin(objectives, axis=0)
    result["posterior_probability_optimal"] = np.bincount(winners, minlength=len(result)) / posterior_draws
    result["objective"] = result.posterior_mean_es_995 + result.annual_cost; result["tail_risk_index"] = result.posterior_mean_es_995
    result["selected"] = False; result.loc[result.loc[feasible, "posterior_probability_optimal"].idxmax(), "selected"] = True
    result["pareto"] = result.apply(lambda r: not ((result.annual_cost <= r.annual_cost) & (result.posterior_mean_es_995 < r.posterior_mean_es_995)).any(), axis=1)
    return result.sort_values(["selected", "posterior_probability_optimal", "objective"], ascending=[False, False, True]).reset_index(drop=True)
