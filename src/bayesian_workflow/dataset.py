from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .states import STAGE_ORDER, WorkflowState


def build_transition_observations(
    runs_path: str | Path,
    stages_path: str | Path,
    output_path: str | Path | None = None,
    materiality: float = 0.01,
) -> pd.DataFrame:
    """Create one observable multistate record per run and workflow stage.

    States are observable here because fault injection, paired clean paths,
    detection, recovery and hard failures are logged. This controlled
    construction is not production telemetry.
    """
    runs = pd.read_parquet(runs_path)
    stages = pd.read_parquet(stages_path)
    run_cols = [
        "run_id", "detected", "detection_stage", "hard_failure", "silent_failure",
        "token_cost", "analyst_minutes", "paired_final_error",
    ]
    frame = stages.merge(runs[run_cols], on="run_id", how="left", validate="many_to_one")
    frame = frame.sort_values(["run_id", "stage_order"]).reset_index(drop=True)
    injection_order = frame["injection_stage"].map(STAGE_ORDER).fillna(99).astype(int)
    detection_order = frame["detection_stage"].map(STAGE_ORDER).fillna(99).astype(int)
    after: list[str] = []
    for row, inject_at, detect_at in zip(frame.itertuples(), injection_order, detection_order):
        if bool(row.hard_failure) and row.stage_order == 6:
            state = WorkflowState.HARD_FAILURE
        elif row.fault_type == "CONTROL" or row.stage_order < inject_at:
            state = WorkflowState.CLEAN
        elif bool(row.detected) and row.stage_order >= detect_at:
            state = WorkflowState.RECOVERED
        else:
            state = WorkflowState.ERROR
        after.append(str(state))
    frame["state_after"] = after
    frame["state_before"] = frame.groupby("run_id")["state_after"].shift().fillna(str(WorkflowState.CLEAN))
    frame["recovered"] = (frame["state_after"] == str(WorkflowState.RECOVERED)) & (
        frame["state_before"] != str(WorkflowState.RECOVERED)
    )
    frame["final_material_failure"] = frame["paired_final_error"].abs() >= materiality
    frame["paired_log_magnitude"] = np.log(frame["paired_delta"].abs() + 1e-9)
    frame["paired_log_increment"] = frame.groupby("run_id")["paired_log_magnitude"].diff()
    columns = [
        "run_id", "world_id", "topology", "fault_type", "injection_stage", "stage",
        "stage_order", "state_before", "state_after", "detected", "recovered",
        "hard_failure", "paired_delta", "paired_relative_delta", "paired_log_magnitude",
        "paired_log_increment", "lineage_completeness", "token_cost", "analyst_minutes",
        "final_material_failure",
    ]
    result = frame[columns].copy()
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(destination, index=False)
    return result
