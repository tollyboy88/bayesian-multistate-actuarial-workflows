import numpy as np
import pandas as pd

from .transition_model import HierarchicalDirichletTransitionModel, STATES
from .priors import DEFAULT_PRIORS


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


def held_out_fault_cell_scores(observations: pd.DataFrame) -> pd.DataFrame:
    """Leave each fault class out and score its transitions using hierarchical borrowing."""
    rows = []
    for fault in sorted(observations.fault_type.unique()):
        train = observations.loc[observations.fault_type != fault]
        test = observations.loc[observations.fault_type == fault]
        model = HierarchicalDirichletTransitionModel().fit(train)
        losses, briers = [], []
        for row in test.itertuples():
            p = np.clip(model.mean(row.stage, row.fault_type, row.topology, row.state_before), 1e-9, 1.0)
            p /= p.sum(); y = STATES.index(row.state_after)
            losses.append(-np.log(p[y])); target = np.eye(len(STATES))[y]
            briers.append(np.square(p - target).sum())
        rows.append({"held_out_fault": fault, "n": len(test), "log_score": np.mean(losses), "brier_score": np.mean(briers)})
    return pd.DataFrame(rows)


def prior_sensitivity(observations: pd.DataFrame) -> pd.DataFrame:
    """Quantify prior sensitivity for the observed E-to-E transition cells."""
    rows = []
    for name, spec in DEFAULT_PRIORS.items():
        strength = float(spec["transition_strength"])
        model = HierarchicalDirichletTransitionModel(
            concentration_shape=max(strength / 2.0, 0.5), concentration_rate=0.5
        ).fit(observations)
        values = []
        for stage, fault, topology in observations[["stage", "fault_type", "topology"]].drop_duplicates().itertuples(index=False):
            values.append(model.mean(stage, fault, topology, "E")[STATES.index("E")])
        rows.append({"prior": name, "mean_e_to_e": np.mean(values), "sd_e_to_e": np.std(values),
            "minimum_e_to_e": np.min(values), "maximum_e_to_e": np.max(values)})
    return pd.DataFrame(rows)
