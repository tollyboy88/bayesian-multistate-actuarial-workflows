from enum import StrEnum


class WorkflowState(StrEnum):
    CLEAN = "C"
    ERROR = "E"
    RECOVERED = "R"
    HARD_FAILURE = "H"


STAGE_ORDER = {
    "S1_INGEST": 1,
    "S2_RECONCILE": 2,
    "S3_SEGMENT": 3,
    "S4_MODEL": 4,
    "S5_SELECT": 5,
    "S6_NARRATE": 6,
}
