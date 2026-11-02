from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bayesian_workflow.diagnostics import held_out_fault_cell_scores, held_out_scores, prior_sensitivity
from bayesian_workflow.optimise import enumerate_stage_controls
from bayesian_workflow.propagation_model import fit_markov_additive
from bayesian_workflow.recursion import terminal_distribution
from bayesian_workflow.simulation_study import run_recovery_study
from bayesian_workflow.transition_model import HierarchicalDirichletTransitionModel, STATES


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transition_observations.parquet"
OUTPUT = ROOT / "outputs"


def write(frame: pd.DataFrame, name: str) -> None:
    target = OUTPUT / "tables" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target.with_suffix(".csv"), index=False)
    frame.to_parquet(target.with_suffix(".parquet"), index=False)


def main() -> None:
    observations = pd.read_parquet(DATA)
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    model = HierarchicalDirichletTransitionModel().fit(observations)
    transitions = model.summary()
    write(transitions, "transition_posterior")
    write(fit_markov_additive(observations), "propagation_posterior")
    scores = held_out_scores(observations, [6, 7])
    write(scores, "held_out_model_comparison")
    fault_scores = held_out_fault_cell_scores(observations)
    write(fault_scores, "held_out_fault_cell_scores")
    sensitivity = prior_sensitivity(observations)
    write(sensitivity, "prior_sensitivity")

    recovery = run_recovery_study()
    write(recovery, "parameter_recovery")
    recovery_summary = recovery.groupby(["parameter", "topology"], as_index=False).agg(
        bias=("estimate", lambda x: float((x - recovery.loc[x.index, "truth"]).mean())),
        rmse=("estimate", lambda x: float(np.sqrt(np.square(x - recovery.loc[x.index, "truth"]).mean()))),
        coverage_90=("covered_90", "mean"),
        coverage_95=("covered_95", "mean"),
    )
    write(recovery_summary, "parameter_recovery_summary")

    optimisation = enumerate_stage_controls(observations)
    write(optimisation, "control_optimisation")

    matrices = []
    for stage in sorted(observations["stage"].unique(), key=lambda value: int(value[1])):
        matrix = np.eye(4)
        matrix[1] = model.mean(stage, "F1_DATA", "T2_VALIDATOR", "E")
        matrices.append(matrix)
    analytic = terminal_distribution(np.array([0.0, 1.0, 0.0, 0.0]), matrices)
    rng = np.random.default_rng(20260923)
    states = np.ones(200_000, dtype=int)
    for matrix in matrices:
        active = states == 1
        states[active] = rng.choice(4, size=active.sum(), p=matrix[1])
    monte_carlo = np.bincount(states, minlength=4) / len(states)
    recursion = pd.DataFrame(
        {
            "state": STATES,
            "analytic": analytic,
            "monte_carlo": monte_carlo,
            "absolute_difference": np.abs(analytic - monte_carlo),
        }
    )
    write(recursion, "recursion_validation")

    active_recovery = transitions.loc[
        (transitions.state_before == "E") & (transitions.state_after == "R")
    ]
    means = active_recovery.groupby("topology").agg(
        mean=("posterior_mean", "mean"), lo=("ci_05", "mean"), hi=("ci_95", "mean")
    ).sort_values("mean")
    plt.figure(figsize=(9, 4.8))
    plt.errorbar(means.index, means["mean"], yerr=[means["mean"] - means["lo"], means["hi"] - means["mean"]], fmt="o", capsize=3)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Posterior recovery probability")
    plt.tight_layout()
    plt.savefig(figures / "figure2_transition_intervals.png", dpi=300)
    plt.close()

    summary_indexed = recovery_summary.loc[recovery_summary.parameter == "E_to_E"].set_index("topology")
    plt.figure(figsize=(6.5, 4.5))
    plt.bar(summary_indexed.index, summary_indexed.coverage_90)
    plt.axhline(0.9, color="black", linestyle="--")
    plt.ylim(0, 1)
    plt.ylabel("Empirical coverage of 90% interval")
    plt.tight_layout()
    plt.savefig(figures / "figure3_parameter_recovery.png", dpi=300)
    plt.close()

    pareto = optimisation.loc[optimisation.pareto].sort_values("annual_cost")
    plt.figure(figsize=(7, 4.5))
    plt.scatter(optimisation.annual_cost, optimisation.tail_risk_index, alpha=0.25)
    plt.plot(pareto.annual_cost, pareto.tail_risk_index, marker="o")
    plt.yscale("log")
    plt.xlabel("Annual control cost index")
    plt.ylabel("Posterior tail-risk index")
    plt.tight_layout()
    plt.savefig(figures / "figure5_pareto_frontier.png", dpi=300)
    plt.close()

    selected = optimisation.loc[optimisation.selected].iloc[0]
    report = {
        "observations": len(observations),
        "worlds": int(observations.world_id.nunique()),
        "hierarchical_log_score": float(scores.loc[scores.model == "hierarchical", "log_score"].iloc[0]),
        "global_log_score": float(scores.loc[scores.model == "global", "log_score"].iloc[0]),
        "mean_90_coverage": float(recovery_summary.coverage_90.mean()),
        "mean_95_coverage": float(recovery_summary.coverage_95.mean()),
        "max_recursion_difference": float(recursion.absolute_difference.max()),
        "selected_placement": str(selected.placement).zfill(6),
        "selected_cost": float(selected.annual_cost),
        "selected_silent_probability": float(selected.posterior_mean_silent_probability),
        "selected_es_995": float(selected.posterior_mean_es_995),
        "selected_probability_optimal": float(selected.posterior_probability_optimal),
        "selected_constraint_probability": float(selected.posterior_probability_risk_constraint),
        "held_out_fault_mean_log_score": float(fault_scores.log_score.mean()),
    }
    (OUTPUT / "RESULTS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {}
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
