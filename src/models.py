"""
Shared data models for FPSIMP.
"""
from enum import Enum

class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    COLABFOLD_RUNNING = "colabfold_running"
    COLABFOLD_COMPLETE = "colabfold_complete"
    SAMPLING_COMPLETE = "sampling_complete"
    COMPLETED = "completed"
    FAILED = "failed"
    COLABFOLD_FAILED = "colabfold_failed" # Added this one as it was used in code
