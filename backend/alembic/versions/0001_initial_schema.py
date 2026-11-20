"""Initial RailVision PostgreSQL schema parity with Flyway V1__initial_schema.sql.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-11-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial_schema"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create core RailVision inspection, job, model version, and wagon tables."""
    op.create_table(
        "inspections",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("stored_filename", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=100), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256_checksum", sa.String(length=64), nullable=False),
        sa.Column("duration_ms", sa.BigInteger(), nullable=True),
        sa.Column("width_px", sa.Integer(), nullable=True),
        sa.Column("height_px", sa.Integer(), nullable=True),
        sa.Column("fps", sa.Double(), nullable=True),
        sa.Column("codec", sa.String(length=64), nullable=True),
        sa.Column("latest_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_inspections"),
        sa.UniqueConstraint("stored_filename", name="inspections_stored_filename_key"),
    )

    op.create_table(
        "processing_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("inspection_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("progress_percent", sa.Integer(), nullable=True),
        sa.Column("engine_type", sa.String(length=64), nullable=False),
        sa.Column("is_demo_simulation", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("error_code", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["inspection_id"],
            ["inspections.id"],
            name="fk_processing_jobs_inspection_id_inspections",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_processing_jobs"),
    )
    op.create_index(
        "idx_processing_jobs_inspection_created",
        "processing_jobs",
        ["inspection_id", sa.text("created_at DESC")],
        unique=False,
    )
    op.create_index(
        "idx_processing_jobs_status",
        "processing_jobs",
        ["status"],
        unique=False,
    )

    op.create_table(
        "model_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("task", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("runtime", sa.String(length=64), nullable=False),
        sa.Column("weights_sha256", sa.String(length=64), nullable=True),
        sa.Column("artifact_ref", sa.String(length=255), nullable=True),
        sa.Column("is_demo_placeholder", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_model_versions"),
        sa.UniqueConstraint("task", "name", "version", name="uq_model_versions_task_name_version"),
    )

    op.create_table(
        "wagon_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("inspection_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("track_id", sa.String(length=64), nullable=False),
        sa.Column("sequence_index", sa.Integer(), nullable=False),
        sa.Column("physical_id", sa.String(length=64), nullable=True),
        sa.Column("material_class", sa.String(length=64), nullable=True),
        sa.Column("material_confidence", sa.Double(), nullable=True),
        sa.Column("material_rejected_as_unknown", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("vote_distribution_json", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("loading_level", sa.String(length=32), nullable=False),
        sa.Column("loading_confidence", sa.Double(), nullable=True),
        sa.Column("loading_spread_json", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("estimated_fill_percentage", sa.Double(), nullable=True),
        sa.Column("measurement_type", sa.String(length=64), server_default=sa.text("'visual_estimate'"), nullable=False),
        sa.Column("physical_volume_verified", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("observations_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("consolidation_method", sa.String(length=64), nullable=True),
        sa.Column("disagreements_json", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("quality_flags_json", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("review_required", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("review_reasons_json", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'[]'::jsonb"), nullable=False),
        sa.Column("model_versions_json", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("is_example", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["inspection_id"],
            ["inspections.id"],
            name="fk_wagon_results_inspection_id_inspections",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["processing_jobs.id"],
            name="fk_wagon_results_job_id_processing_jobs",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_wagon_results"),
    )
    op.create_index(
        "idx_wagon_results_inspection_job_seq",
        "wagon_results",
        ["inspection_id", "job_id", "sequence_index"],
        unique=False,
    )

    op.create_table(
        "wagon_observations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("wagon_result_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("frame_index", sa.Integer(), nullable=False),
        sa.Column("frame_timestamp_ms", sa.BigInteger(), nullable=False),
        sa.Column("bbox_x", sa.Double(), nullable=False),
        sa.Column("bbox_y", sa.Double(), nullable=False),
        sa.Column("bbox_width", sa.Double(), nullable=False),
        sa.Column("bbox_height", sa.Double(), nullable=False),
        sa.Column("coordinate_system", sa.String(length=32), server_default=sa.text("'normalized'"), nullable=False),
        sa.Column("detected_class", sa.String(length=64), nullable=False),
        sa.Column("detection_confidence", sa.Double(), nullable=False),
        sa.Column("material_class", sa.String(length=64), nullable=True),
        sa.Column("material_confidence", sa.Double(), nullable=True),
        sa.Column("loading_level", sa.String(length=32), nullable=True),
        sa.Column("loading_confidence", sa.Double(), nullable=True),
        sa.Column("is_example", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.ForeignKeyConstraint(
            ["wagon_result_id"],
            ["wagon_results.id"],
            name="fk_wagon_observations_wagon_result_id_wagon_results",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_wagon_observations"),
    )
    op.create_index(
        "idx_wagon_observations_result_ts",
        "wagon_observations",
        ["wagon_result_id", "frame_timestamp_ms"],
        unique=False,
    )


def downgrade() -> None:
    """Drop RailVision schema objects in reverse dependency order."""
    op.drop_index("idx_wagon_observations_result_ts", table_name="wagon_observations")
    op.drop_table("wagon_observations")

    op.drop_index("idx_wagon_results_inspection_job_seq", table_name="wagon_results")
    op.drop_table("wagon_results")

    op.drop_table("model_versions")

    op.drop_index("idx_processing_jobs_status", table_name="processing_jobs")
    op.drop_index("idx_processing_jobs_inspection_created", table_name="processing_jobs")
    op.drop_table("processing_jobs")

    op.drop_table("inspections")
