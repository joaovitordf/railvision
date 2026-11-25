"""Pydantic DTOs for managed YOLO detector training runs and versioned lineage."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator

from app.core.schemas import CamelModel

TrainingRunStatus = Literal["QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]

SUPPORTED_TRAINING_IMGSZ = {512, 640, 768, 960, 1024}


class TrainingDatasetOptionDto(CamelModel):
    """Selectable YOLO dataset discovered under data/processed."""

    id: str
    name: str
    data_yaml_path: str
    train_count: int
    val_count: int
    val_campaigns: list[str] = Field(default_factory=list)
    val_fingerprint: str
    has_frozen_test: bool = False
    warning: str | None = None


class TrainingBaseOptionDto(CamelModel):
    """Selectable starting checkpoint or pretrained architecture for advanced overrides."""

    id: str
    label: str
    kind: Literal["auto", "run", "checkpoint", "pretrained"]
    description: str


class TrainingEpochPointDto(CamelModel):
    """Per-epoch training & validation metrics parsed live from Ultralytics results.csv."""

    epoch: int
    time_sec: float
    train_box_loss: float
    train_cls_loss: float
    train_l1_loss: float
    val_box_loss: float
    val_cls_loss: float
    val_l1_loss: float
    precision: float
    recall: float
    map50: float
    map50_95: float
    lr: float | None = None


class TrainingLineagePointDto(CamelModel):
    """Single point along the sequential model evolution timeline (v0 -> v1 -> v2...)."""

    version_label: str
    run_id: str | None = None
    experiment_name: str
    status: str
    dataset_id: str
    val_fingerprint: str | None = None
    val_dataset_changed: bool = False
    base_label: str
    created_at: datetime | None = None
    epochs_completed: int = 0
    map50: float | None = None
    map50_95: float | None = None
    precision: float | None = None
    recall: float | None = None
    best_f1_conf: float | None = None
    registered_model_id: str | None = None
    notes: str | None = None


class TrainingRunSummaryDto(CamelModel):
    """Summary card/row for a managed detector training run."""

    run_id: UUID
    version_number: int
    version_label: str
    experiment_name: str
    status: TrainingRunStatus
    dataset_id: str
    val_fingerprint: str
    base_id: str
    base_label: str
    architecture: str
    epochs_requested: int
    epochs_completed: int = 0
    current_epoch: int = 0
    progress_percent: int = 0
    imgsz: int = 640
    batch: int = 8
    freeze: int = 10
    lr0: float = 0.0005
    cos_lr: bool = True
    patience: int = 15
    best_map50: float | None = None
    best_map50_95: float | None = None
    final_map50: float | None = None
    final_map50_95: float | None = None
    final_precision: float | None = None
    final_recall: float | None = None
    best_f1_conf: float | None = None
    peak_vram_mib: float | None = None
    wall_time_s: float | None = None
    best_weights_path: str | None = None
    registered_model_id: str | None = None
    notes: str | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class CalibrationCurvePointDto(CamelModel):
    """Confidence threshold calibration point computed on the validation split."""

    conf: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


class TrainingRunDetailDto(TrainingRunSummaryDto):
    """Detailed training run payload including live epoch curves and log tail."""

    epoch_history: list[TrainingEpochPointDto] = Field(default_factory=list)
    log_tail: list[str] = Field(default_factory=list)
    val_calibration_curve: list[CalibrationCurvePointDto] = Field(default_factory=list)


class TrainingBaselineMetricsDto(CamelModel):
    """Fixed baseline metrics (v0 / active checkpoint) shown as chart reference."""

    model_id: str
    version_label: str = "v0"
    source_experiment: str
    map50: float | None = None
    map50_95: float | None = None
    precision: float | None = None
    recall: float | None = None


class TrainingOverviewDto(CamelModel):
    """Complete state of the sequential Training Lineage panel."""

    lineage_name: str = "Detector de Vagões Abertos"
    baseline_model_id: str
    baseline_metrics: TrainingBaselineMetricsDto
    next_version_number: int
    next_version_label: str
    recommended_base_id: str
    recommended_base_label: str
    active_run_id: UUID | None = None
    gpu_available: bool
    gpu_name: str | None = None
    datasets: list[TrainingDatasetOptionDto] = Field(default_factory=list)
    lineage_points: list[TrainingLineagePointDto] = Field(default_factory=list)
    runs: list[TrainingRunSummaryDto] = Field(default_factory=list)
    advanced_base_options: list[TrainingBaseOptionDto] = Field(default_factory=list)


class StartTrainingRequest(CamelModel):
    """Request to launch a new incremental version of the open-top wagon detector."""

    dataset_id: str = Field(default="open-top-yolo-v1", min_length=2, max_length=120)
    base_override: str | None = Field(default=None, max_length=160)
    epochs: int = Field(default=30, ge=1, le=200)
    patience: int = Field(default=15, ge=3, le=100)
    imgsz: int = Field(default=640)
    batch: int = Field(default=8, ge=1, le=64)
    freeze: int = Field(default=10, ge=0, le=30)
    lr0: float = Field(default=0.0005, gt=0.0, le=0.1)
    cos_lr: bool = True
    notes: str | None = Field(default=None, max_length=500)

    @field_validator("imgsz")
    @classmethod
    def validate_imgsz(cls, value: int) -> int:
        if value not in SUPPORTED_TRAINING_IMGSZ:
            allowed = ", ".join(str(x) for x in sorted(SUPPORTED_TRAINING_IMGSZ))
            raise ValueError(f"imgsz inválido ({value}). Valores suportados: {allowed}.")
        return value


class RegisterTrainingCheckpointRequest(CamelModel):
    """Optional overrides when promoting a completed training run into Model Registry."""

    model_id: str | None = Field(default=None, min_length=3, max_length=120, pattern=r"^[a-zA-Z0-9._-]+$")
    activate: bool = False
