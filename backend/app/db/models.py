"""SQLAlchemy ORM models matching V1__initial_schema.sql in PostgreSQL."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Inspection(Base):
    """Inspection media item and lifecycle state."""

    __tablename__ = "inspections"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    title: Mapped[str] = mapped_column(sa.String(200), nullable=False)
    notes: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    status: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    original_filename: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    stored_filename: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(sa.BigInteger, nullable=False)
    sha256_checksum: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True)
    width_px: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    height_px: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(sa.Double, nullable=True)
    codec: Mapped[str | None] = mapped_column(sa.String(64), nullable=True)
    latest_job_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)

    __table_args__ = (
        sa.UniqueConstraint("stored_filename", name="inspections_stored_filename_key"),
        sa.Index("idx_inspections_created_at_desc", created_at.desc()),
    )

    jobs: Mapped[list["ProcessingJob"]] = relationship(
        back_populates="inspection",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    wagon_results: Mapped[list["WagonResult"]] = relationship(
        back_populates="inspection",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ProcessingJob(Base):
    """Asynchronous processing execution linked to an inspection."""

    __tablename__ = "processing_jobs"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    inspection_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        sa.ForeignKey("inspections.id", ondelete="CASCADE"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    progress_percent: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    engine_type: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    is_demo_simulation: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("false"), default=False
    )
    error_code: Mapped[str | None] = mapped_column(sa.String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(sa.DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)

    __table_args__ = (
        sa.Index("idx_processing_jobs_inspection_created", inspection_id, created_at.desc()),
        sa.Index("idx_processing_jobs_status", status),
        sa.Index(
            "uq_processing_jobs_one_active_per_inspection",
            inspection_id,
            unique=True,
            postgresql_where=sa.text("status IN ('QUEUED', 'RUNNING')"),
        ),
    )

    inspection: Mapped["Inspection"] = relationship(back_populates="jobs")
    wagon_results: Mapped[list["WagonResult"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ModelVersion(Base):
    """Registered model artifact metadata and runtime version."""

    __tablename__ = "model_versions"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    task: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    name: Mapped[str] = mapped_column(sa.String(120), nullable=False)
    version: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    runtime: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    weights_sha256: Mapped[str | None] = mapped_column(sa.String(64), nullable=True)
    artifact_ref: Mapped[str | None] = mapped_column(sa.String(255), nullable=True)
    is_demo_placeholder: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("false"), default=False
    )
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)

    __table_args__ = (
        sa.UniqueConstraint(
            "task",
            "name",
            "version",
            "runtime",
            name="uq_model_versions_task_name_version_runtime",
        ),
    )


class WagonResult(Base):
    """Consolidated wagon result across frames in an inspection job."""

    __tablename__ = "wagon_results"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    inspection_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        sa.ForeignKey("inspections.id", ondelete="CASCADE"),
        nullable=False,
    )
    job_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        sa.ForeignKey("processing_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    track_id: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    sequence_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    physical_id: Mapped[str | None] = mapped_column(sa.String(64), nullable=True)
    material_class: Mapped[str | None] = mapped_column(sa.String(64), nullable=True)
    material_confidence: Mapped[float | None] = mapped_column(sa.Double, nullable=True)
    material_rejected_as_unknown: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("false"), default=False
    )
    vote_distribution_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'{}'::jsonb"), default=dict
    )
    loading_level: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    loading_confidence: Mapped[float | None] = mapped_column(sa.Double, nullable=True)
    loading_spread_json: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'[]'::jsonb"), default=list
    )
    estimated_fill_percentage: Mapped[float | None] = mapped_column(sa.Double, nullable=True)
    measurement_type: Mapped[str] = mapped_column(
        sa.String(64),
        nullable=False,
        server_default=sa.text("'visual_estimate'"),
        default="visual_estimate",
    )
    physical_volume_verified: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("false"), default=False
    )
    observations_count: Mapped[int] = mapped_column(
        sa.Integer, nullable=False, server_default=sa.text("0"), default=0
    )
    consolidation_method: Mapped[str | None] = mapped_column(sa.String(64), nullable=True)
    disagreements_json: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'[]'::jsonb"), default=list
    )
    quality_flags_json: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'[]'::jsonb"), default=list
    )
    review_required: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("false"), default=False
    )
    review_reasons_json: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'[]'::jsonb"), default=list
    )
    model_versions_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=sa.text("'{}'::jsonb"), default=dict
    )
    is_example: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("false"), default=False
    )
    created_at: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)

    __table_args__ = (
        sa.Index("idx_wagon_results_inspection_job_seq", inspection_id, job_id, sequence_index.asc()),
    )

    inspection: Mapped["Inspection"] = relationship(back_populates="wagon_results")
    job: Mapped["ProcessingJob"] = relationship(back_populates="wagon_results")
    observations: Mapped[list["WagonObservation"]] = relationship(
        back_populates="wagon_result",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class WagonObservation(Base):
    """Per-frame detection observation linked to a consolidated wagon result."""

    __tablename__ = "wagon_observations"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    wagon_result_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        sa.ForeignKey("wagon_results.id", ondelete="CASCADE"),
        nullable=False,
    )
    frame_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    frame_timestamp_ms: Mapped[int] = mapped_column(sa.BigInteger, nullable=False)
    bbox_x: Mapped[float] = mapped_column(sa.Double, nullable=False)
    bbox_y: Mapped[float] = mapped_column(sa.Double, nullable=False)
    bbox_width: Mapped[float] = mapped_column(sa.Double, nullable=False)
    bbox_height: Mapped[float] = mapped_column(sa.Double, nullable=False)
    coordinate_system: Mapped[str] = mapped_column(
        sa.String(32),
        nullable=False,
        server_default=sa.text("'normalized'"),
        default="normalized",
    )
    detected_class: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    detection_confidence: Mapped[float] = mapped_column(sa.Double, nullable=False)
    material_class: Mapped[str | None] = mapped_column(sa.String(64), nullable=True)
    material_confidence: Mapped[float | None] = mapped_column(sa.Double, nullable=True)
    loading_level: Mapped[str | None] = mapped_column(sa.String(32), nullable=True)
    loading_confidence: Mapped[float | None] = mapped_column(sa.Double, nullable=True)
    is_example: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, server_default=sa.text("false"), default=False
    )

    __table_args__ = (
        sa.Index("idx_wagon_observations_result_ts", wagon_result_id, frame_timestamp_ms.asc()),
    )

    wagon_result: Mapped["WagonResult"] = relationship(back_populates="observations")
