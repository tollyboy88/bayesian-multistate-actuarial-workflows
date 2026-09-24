import numpy as np
import pandas as pd

from .transition_model import HierarchicalDirichletTransitionModel


def run_recovery_study(replications: int = 80, n_per_cell: int = 60, seed: int = 20260923) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    truth = {"T1_LINEAR": np.array([0.02, 0.93, 0.04, 0.01]), "T2_VALIDATOR": np.array([0.02, 0.20, 0.77, 0.01])}
    rows = []
    for rep in range(replications):
        data = []
        for topology, p in truth.items():
            outcomes = rng.choice(["C", "E", "R", "H"], size=n_per_cell, p=p)
            for outcome in outcomes:
                data.append({"stage":"S3_SEGMENT", "fault_type":"F1_DATA", "topology":topology, "state_before":"E", "state_after":outcome})
        model = HierarchicalDirichletTransitionModel(prior_strength=0.5).fit(pd.DataFrame(data))
        for topology, p in truth.items():
            draws = rng.dirichlet(model.alpha("S3_SEGMENT", "F1_DATA", topology, "E"), size=2_000)[:, 1]
            lo90, hi90 = np.quantile(draws, [0.05, 0.95])
            lo95, hi95 = np.quantile(draws, [0.025, 0.975]); estimate = draws.mean()
            rows.append({"replication":rep, "topology":topology, "parameter":"E_to_E", "truth":p[1], "estimate":estimate, "ci_05":lo90, "ci_95":hi90, "ci_025":lo95, "ci_975":hi95, "covered_90":lo90 <= p[1] <= hi90, "covered_95":lo95 <= p[1] <= hi95})
    return pd.DataFrame(rows)
