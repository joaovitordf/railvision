"""WS0 DB hardening: single active job constraint, runtime-aware model key, and inspections index.

Revision ID: 0002_hardening_constraints
Revises: 0001_initial_schema
Create Date: 2026-11-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_hardening_constraints"
down_revision: str | Sequence[str] | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply WS0 uniqueness and query index constraints safely over existing V1 schema."""
    op.create_index(
        "idx_inspections_created_at_desc",
        "inspections",
        [sa.text("created_at DESC")],
        unique=False,
    )
    op.create_index(
        "uq_processing_jobs_one_active_per_inspection",
        "processing_jobs",
        ["inspection_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('QUEUED', 'RUNNING')"),
    )
    op.drop_constraint("uq_model_versions_task_name_version", "model_versions", type_="unique")
    op.create_unique_constraint(
        "uq_model_versions_task_name_version_runtime",
        "model_versions",
        ["task", "name", "version", "runtime"],
    )


def downgrade() -> None:
    """Revert WS0 constraints back to 0001_initial_schema."""
    op.drop_constraint("uq_model_versions_task_name_version_runtime", "model_versions", type_="unique")
    op.create_unique_constraint(
        "uq_model_versions_task_name_version",
        "model_versions",
        ["task", "name", "version"],
    )
    op.drop_index("uq_processing_jobs_one_active_per_inspection", table_name="processing_jobs")
    op.drop_index("idx_inspections_created_at_desc", table_name="inspections")
