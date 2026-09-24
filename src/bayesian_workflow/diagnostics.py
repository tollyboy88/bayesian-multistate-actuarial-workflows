import numpy as np
import pandas as pd

from .transition_model import HierarchicalDirichletTransitionModel, STATES


def held_out_scores(observations: pd.DataFrame, holdout_worlds: list[int]) -> pd.DataFrame:
    train = observations.loc[~observations["world_id"].isin(holdout_worlds)]
    test = observations.loc[observations["world_id"].isin(holdout_worlds)]
    model = HierarchicalDirichletTransitionModel().fit(train)
    global_probs = train["state_after"].value_counts(normalize=True).reindex(STATES, fill_value=1e-9)
    rows = []
    for name in ["global", "hierarchical"]:
        probs, outcomes = [], []
        for row in test.itertuples():
            p = global_probs.to_numpy(float) if name == "global" else model.mean(row.stage, row.fault_type, row.topology, row.state_before)
            p = np.clip(p, 1e-9, 1); p = p / p.sum()
            y = STATES.index(row.state_after)
            probs.append(p); outcomes.append(y)
        matrix = np.asarray(probs); y = np.asarray(outcomes); one_hot = np.eye(len(STATES))[y]
        rows.append({
            "model": name, "held_out_worlds": ",".join(map(str, holdout_worlds)), "n": len(y),
            "log_score": float(-np.log(matrix[np.arange(len(y)), y]).mean()),
            "brier_score": float(np.square(matrix - one_hot).sum(axis=1).mean()),
        })
    return pd.DataFrame(rows)
