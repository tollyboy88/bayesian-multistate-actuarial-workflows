import numpy as np
import pandas as pd
from .transition_model import HierarchicalDirichletTransitionModel, STATES


def run_recovery_study(replications: int = 40, n_per_cell: int = 50, seed: int = 20260923) -> pd.DataFrame:
    """Recover multiple transition and additive-severity parameters across 27 cells."""
    rng = np.random.default_rng(seed)
    stages = ["S2_RECONCILE", "S4_MODEL", "S6_REPORT"]
    faults = ["F1_DATA", "F4_JUDGEMENT", "F7_CONTEXT"]
    topologies = ["T1_LINEAR", "T2_VALIDATOR", "T3_SUPERVISOR"]
    base_e = {"T1_LINEAR": .82, "T2_VALIDATOR": .28, "T3_SUPERVISOR": .42}
    rows = []
    for rep in range(replications):
        data = []; severity_truth = {}
        for si, stage in enumerate(stages):
            for fi, fault in enumerate(faults):
                for topology in topologies:
                    p_e = np.clip(base_e[topology] + .025 * si + .015 * fi, .05, .92)
                    p = np.array([.03, p_e, .94 - p_e, .03])
                    outcomes = rng.choice(STATES, n_per_cell, p=p)
                    mu = .08 * si + .12 * fi + (0.25 if topology == "T1_LINEAR" else -0.05)
                    severity_truth[(stage, fault, topology)] = mu
                    increments = rng.normal(mu, .45, n_per_cell)
                    for outcome, increment in zip(outcomes, increments):
                        data.append({"stage": stage, "fault_type": fault, "topology": topology,
                            "state_before": "E", "state_after": outcome, "paired_log_increment": increment})
        frame = pd.DataFrame(data); model = HierarchicalDirichletTransitionModel().fit(frame)
        for stage in stages:
            for fault in faults:
                for topology in topologies:
                    key = (stage, fault, topology); draws = model.draw_many(*key, "E", 800, rng)
                    for parameter, index in [("E_to_E", 1), ("E_to_R", 2), ("E_to_H", 3)]:
                        truth = {"E_to_E": base_e[topology] + .025 * stages.index(stage) + .015 * faults.index(fault),
                            "E_to_R": .94 - (base_e[topology] + .025 * stages.index(stage) + .015 * faults.index(fault)), "E_to_H": .03}[parameter]
                        values = draws[:, index]; lo90, hi90 = np.quantile(values, [.05, .95]); lo95, hi95 = np.quantile(values, [.025, .975])
                        rows.append({"replication": rep, "stage": stage, "fault_type": fault, "topology": topology,
                            "parameter": parameter, "truth": truth, "estimate": values.mean(), "ci_05": lo90, "ci_95": hi90,
                            "ci_025": lo95, "ci_975": hi95, "covered_90": lo90 <= truth <= hi90, "covered_95": lo95 <= truth <= hi95})
                    x = frame.loc[(frame.stage == stage) & (frame.fault_type == fault) & (frame.topology == topology), "paired_log_increment"].to_numpy()
                    estimate = x.mean(); se = x.std(ddof=1) / np.sqrt(len(x)); lo90, hi90 = estimate - 1.645 * se, estimate + 1.645 * se; lo95, hi95 = estimate - 1.96 * se, estimate + 1.96 * se
                    truth = severity_truth[key]
                    rows.append({"replication": rep, "stage": stage, "fault_type": fault, "topology": topology,
                        "parameter": "additive_increment", "truth": truth, "estimate": estimate, "ci_05": lo90, "ci_95": hi90,
                        "ci_025": lo95, "ci_975": hi95, "covered_90": lo90 <= truth <= hi90, "covered_95": lo95 <= truth <= hi95})
    return pd.DataFrame(rows)
