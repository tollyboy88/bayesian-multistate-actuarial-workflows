import numpy as np
import pandas as pd

from bayesian_workflow.optimise import enumerate_stage_controls
from bayesian_workflow.recursion import terminal_distribution
from bayesian_workflow.transition_model import HierarchicalDirichletTransitionModel


def test_transition_probabilities_sum_to_one():
    frame = pd.DataFrame(
        {
            "stage": ["S1_INGEST"] * 4,
            "fault_type": ["F1_DATA"] * 4,
            "topology": ["T1_LINEAR"] * 4,
            "state_before": ["E"] * 4,
            "state_after": ["C", "E", "R", "H"],
        }
    )
    model = HierarchicalDirichletTransitionModel().fit(frame)
    assert np.isclose(model.mean("S1_INGEST", "F1_DATA", "T1_LINEAR", "E").sum(), 1.0)


def test_terminal_recursion_preserves_probability():
    matrix = np.eye(4)
    matrix[1] = [0.1, 0.3, 0.5, 0.1]
    terminal = terminal_distribution(np.array([0.0, 1.0, 0.0, 0.0]), [matrix, matrix])
    assert np.isclose(terminal.sum(), 1.0)


def test_control_catalogue_has_global_selection():
    rows = []
    for stage in [f"S{i}_STAGE" for i in range(1, 7)]:
        for topology, outcome in [("T1_LINEAR", "E"), ("T2_VALIDATOR", "R")]:
            rows.append({"stage": stage, "topology": topology, "state_before": "E", "state_after": outcome})
    result = enumerate_stage_controls(pd.DataFrame(rows))
    assert len(result) == 64
    assert result.selected.sum() == 1
    assert result.loc[result.selected, "feasible"].all()
