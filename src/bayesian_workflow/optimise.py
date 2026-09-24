from itertools import product

import numpy as np
import pandas as pd


def enumerate_stage_controls(observations: pd.DataFrame, stage_cost: float = 400.0, budget: float = 1_200.0) -> pd.DataFrame:
    """Enumerate all 64 placements of a validator across six stages."""
    stages = sorted(observations["stage"].unique(), key=lambda s: int(s[1]))
    active = observations.loc[observations["state_before"] == "E"]
    rows = []
    for placement in product([0, 1], repeat=len(stages)):
        persistence = 1.0
        for stage, controlled in zip(stages, placement):
            topology = "T2_VALIDATOR" if controlled else "T1_LINEAR"
            cell = active.loc[(active["stage"] == stage) & (active["topology"] == topology)]
            p = float((cell["state_after"] == "E").mean()) if len(cell) else (0.2 if controlled else 0.9)
            persistence *= np.clip(p, 1e-4, 1.0)
        cost = stage_cost * sum(placement); risk_index = persistence * 1_000_000
        rows.append({
            "placement": "".join(map(str, placement)), "controls": sum(placement), "annual_cost": cost,
            "posterior_mean_silent_probability": persistence, "tail_risk_index": risk_index,
            "feasible": cost <= budget, "objective": risk_index + cost,
        })
    result = pd.DataFrame(rows).sort_values(["objective", "annual_cost"]).reset_index(drop=True)
    result["selected"] = False
    feasible = result.index[result["feasible"]]
    if len(feasible): result.loc[feasible[0], "selected"] = True
    result["pareto"] = result.apply(lambda r: not ((result["annual_cost"] <= r["annual_cost"]) & (result["tail_risk_index"] < r["tail_risk_index"])).any(), axis=1)
    return result
