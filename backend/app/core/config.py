"""Typed application settings loaded from environment variables and the repository .env file."""

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.paths import REPO_ROOT, resolve_repo_path


class Settings(BaseSettings):
    """Runtime settings aligned with .env.example and Java application.yml defaults."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    postgres_host: str = Field(
        default="127.0.0.1",
        validation_alias=AliasChoices("POSTGRES_HOST", "postgres_host"),
    )
    postgres_port: int = Field(
        default=5432,
        validation_alias=AliasChoices("POSTGRES_PORT", "postgres_port"),
    )
    postgres_db: str = Field(
        default="railvision",
        validation_alias=AliasChoices("POSTGRES_DB", "postgres_db"),
    )
    postgres_user: str = Field(
        default="railvision",
        validation_alias=AliasChoices("POSTGRES_USER", "postgres_user"),
    )
    postgres_password: str = Field(
        default="",
        repr=False,
        validation_alias=AliasChoices("POSTGRES_PASSWORD", "postgres_password"),
    )
    database_url_override: str | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("RAILVISION_DATABASE_URL", "DATABASE_URL", "database_url"),
    )

    server_port: int = Field(
        default=8080,
        validation_alias=AliasChoices("RAILVISION_SERVER_PORT", "server_port"),
    )
    cors_allowed_origin: str = Field(
        default="http://localhost:4200",
        validation_alias=AliasChoices("RAILVISION_CORS_ALLOWED_ORIGIN", "cors_allowed_origin"),
    )
    storage_root_raw: str = Field(
        default="./data/processed/app-storage",
        validation_alias=AliasChoices("RAILVISION_STORAGE_ROOT", "storage_root"),
    )
    upload_max_size_mb: int = Field(
        default=500,
        validation_alias=AliasChoices("RAILVISION_UPLOAD_MAX_SIZE_MB", "upload_max_size_mb"),
    )

    processing_demo_delay_ms: int = Field(
        default=800,
        validation_alias=AliasChoices("RAILVISION_PROCESSING_DEMO_DELAY_MS", "processing_demo_delay_ms"),
    )
    processing_max_concurrency: int = Field(
        default=1,
        validation_alias=AliasChoices("RAILVISION_PROCESSING_MAX_CONCURRENCY", "processing_max_concurrency"),
    )

    default_runtime: str = Field(
        default="pytorch_cuda",
        validation_alias=AliasChoices("RAILVISION_DEFAULT_RUNTIME", "default_runtime"),
    )
    pytorch_weights_path_raw: str = Field(
        default="artifacts/runs/detect/exp04c_yolo26n_opentop_freeze10/weights/best.pt",
        validation_alias=AliasChoices("RAILVISION_PYTORCH_MODEL_PATH", "pytorch_weights_path"),
    )
    onnx_model_path_raw: str = Field(
        default="artifacts/models/onnx/exp04c_yolo26n_opentop_640.onnx",
        validation_alias=AliasChoices("RAILVISION_ONNX_MODEL_PATH", "onnx_model_path"),
    )
    onnx_model_metadata_path_raw: str = Field(
        default="artifacts/models/onnx/exp04c_yolo26n_opentop_640.meta.json",
        validation_alias=AliasChoices("RAILVISION_ONNX_META_PATH", "onnx_model_metadata_path"),
    )
    onnx_runs_storage_path_raw: str = Field(
        default="data/processed/onnx-spike-runs",
        validation_alias=AliasChoices("RAILVISION_ONNX_RUNS_ROOT", "onnx_runs_storage_path"),
    )
    model_registry_path_raw: str = Field(
        default="artifacts/models/registry.json",
        validation_alias=AliasChoices("RAILVISION_MODEL_REGISTRY_PATH", "model_registry_path"),
    )
    curation_inbox_path_raw: str = Field(
        default="data/processed/curation-inbox",
        validation_alias=AliasChoices("RAILVISION_CURATION_INBOX_ROOT", "curation_inbox_path"),
    )
    annotation_studio_path_raw: str = Field(
        default="data/processed/annotation-studio",
        validation_alias=AliasChoices("RAILVISION_ANNOTATION_STUDIO_ROOT", "annotation_studio_path"),
    )
    training_runs_path_raw: str = Field(
        default="artifacts/runs/training",
        validation_alias=AliasChoices("RAILVISION_TRAINING_RUNS_ROOT", "training_runs_path"),
    )
    datasets_root_raw: str = Field(
        default="data/processed",
        validation_alias=AliasChoices("RAILVISION_DATASETS_ROOT", "datasets_root"),
    )
    default_confidence_threshold: float = Field(
        default=0.25,
        validation_alias=AliasChoices("RAILVISION_DEFAULT_CONFIDENCE_THRESHOLD", "default_confidence_threshold"),
    )
    default_iou_threshold: float = Field(
        default=0.45,
        validation_alias=AliasChoices("RAILVISION_DEFAULT_IOU_THRESHOLD", "default_iou_threshold"),
    )
    max_video_frames_per_ui_run: int = Field(
        default=900,
        validation_alias=AliasChoices("RAILVISION_MAX_VIDEO_FRAMES_PER_UI_RUN", "max_video_frames_per_ui_run"),
    )
    output_video_max_height: int = Field(
        default=1080,
        validation_alias=AliasChoices("RAILVISION_OUTPUT_VIDEO_MAX_HEIGHT", "output_video_max_height"),
    )
    min_track_observations: int = Field(
        default=2,
        validation_alias=AliasChoices("RAILVISION_MIN_TRACK_OBSERVATIONS", "min_track_observations"),
    )

    @property
    def cors_allowed_origins(self) -> list[str]:
        """Parse comma-separated CORS origins preserving single-origin defaults."""
        origins = [item.strip() for item in self.cors_allowed_origin.split(",") if item.strip()]
        return origins or ["http://localhost:4200"]

    @property
    def database_url(self) -> str:
        """Build the PostgreSQL SQLAlchemy/psycopg3 connection URL."""
        if self.database_url_override:
            return self.database_url_override
        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        return (
            f"postgresql+psycopg://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def storage_root(self) -> Path:
        """Resolved root directory for uploaded inspection media."""
        return resolve_repo_path(self.storage_root_raw)

    @property
    def pytorch_weights_path(self) -> Path:
        """Resolved default PyTorch checkpoint path."""
        return resolve_repo_path(self.pytorch_weights_path_raw)

    @property
    def onnx_model_path(self) -> Path:
        """Resolved default ONNX model path."""
        return resolve_repo_path(self.onnx_model_path_raw)

    @property
    def onnx_model_metadata_path(self) -> Path:
        """Resolved default ONNX metadata JSON path."""
        return resolve_repo_path(self.onnx_model_metadata_path_raw)

    @property
    def onnx_runs_storage_path(self) -> Path:
        """Resolved directory for lab inference runs."""
        return resolve_repo_path(self.onnx_runs_storage_path_raw)

    @property
    def model_registry_path(self) -> Path:
        """Resolved path to declarative Model Registry JSON."""
        return resolve_repo_path(self.model_registry_path_raw)

    @property
    def curation_inbox_path(self) -> Path:
        """Resolved directory for HITL frame curation exports."""
        return resolve_repo_path(self.curation_inbox_path_raw)

    @property
    def annotation_studio_path(self) -> Path:
        """Resolved isolated workspace for the native Annotation Studio."""
        return resolve_repo_path(self.annotation_studio_path_raw)

    @property
    def training_runs_path(self) -> Path:
        """Resolved directory for UI-managed detector training runs."""
        return resolve_repo_path(self.training_runs_path_raw)

    @property
    def datasets_root(self) -> Path:
        """Resolved root directory containing processed YOLO datasets."""
        return resolve_repo_path(self.datasets_root_raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
