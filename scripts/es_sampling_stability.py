from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from bayesian_workflow.optimise import enumerate_stage_controls


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "transition_observations.parquet"
TABLES = ROOT / "outputs" / "tables"


def main() -> None:
    observations = pd.read_parquet(DATA)
    sparse = enumerate_stage_controls(observations, samples_per_draw=1_000)
    dense = enumerate_stage_controls(observations, samples_per_draw=10_000)

    columns = [
        "placement",
        "posterior_mean_es_995",
        "posterior_probability_optimal",
        "feasible",
        "selected",
    ]
    comparison = sparse[columns].merge(
        dense[columns], on="placement", suffixes=("_1000", "_10000")
    )
    comparison["es_rank_1000"] = comparison["posterior_mean_es_995_1000"].rank()
    comparison["es_rank_10000"] = comparison["posterior_mean_es_995_10000"].rank()
    comparison["absolute_rank_change"] = (
        comparison["es_rank_10000"] - comparison["es_rank_1000"]
    ).abs()

    feasible = comparison["feasible_10000"]
    top_10_sparse = set(comparison.nsmallest(10, "posterior_mean_es_995_1000")["placement"])
    top_10_dense = set(comparison.nsmallest(10, "posterior_mean_es_995_10000")["placement"])
    summary = {
        "posterior_draws": 300,
        "sparse_samples_per_draw": 1_000,
        "dense_samples_per_draw": 10_000,
        "dense_expected_tail_observations_per_draw": 50,
        "selected_placement_1000": str(sparse.loc[sparse.selected, "placement"].iloc[0]).zfill(6),
        "selected_placement_10000": str(dense.loc[dense.selected, "placement"].iloc[0]).zfill(6),
        "spearman_all_placements": float(
            comparison["posterior_mean_es_995_1000"].corr(
                comparison["posterior_mean_es_995_10000"], method="spearman"
            )
        ),
        "spearman_feasible_placements": float(
            comparison.loc[feasible, "posterior_mean_es_995_1000"].corr(
                comparison.loc[feasible, "posterior_mean_es_995_10000"], method="spearman"
            )
        ),
        "top_10_overlap": len(top_10_sparse & top_10_dense),
        "maximum_absolute_rank_change": int(comparison["absolute_rank_change"].max()),
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(TABLES / "es_sampling_stability.csv", index=False)
    (ROOT / "outputs" / "es_sampling_stability.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    manifest = {}
    for path in sorted((ROOT / "outputs").rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT / "outputs" / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
