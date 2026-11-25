"""Stage M0 verification tests: config, CamelModel, ORM parity, FastAPI /health, PyAV, and Alembic."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from alembic import command
from alembic.config import Config as AlembicConfig
import av
from fastapi.testclient import TestClient
import pytest
import sqlalchemy as sa
import torch

from app.core.config import Settings, get_settings
from app.core.paths import BACKEND_ROOT, REPO_ROOT, resolve_repo_path
from app.core.schemas import CamelModel
from app.db import Base, Inspection, ModelVersion, ProcessingJob, WagonObservation, WagonResult
from app.main import create_app


def _alembic_config(schema: str, database_url: str) -> AlembicConfig:
    cfg = AlembicConfig(str(BACKEND_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    cfg.attributes["schema"] = schema
    cfg.attributes["database_url"] = database_url
    return cfg


def test_settings_and_paths_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings loads .env defaults and resolves both ./data and Java ../artifacts conventions."""
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.postgres_db == "railvision"
    assert settings.postgres_user == "railvision"
    assert settings.postgres_port == 5432
    assert settings.server_port == 8080
    assert settings.cors_allowed_origin == "http://localhost:4200"
    assert settings.default_runtime == "pytorch_cuda"
    assert settings.output_video_max_height == 1080
    assert settings.pytorch_weights_path == (
        REPO_ROOT / "artifacts/runs/detect/exp04c_yolo26n_opentop_freeze10/weights/best.pt"
    ).resolve()
    assert settings.pytorch_weights_path.is_file()
    assert settings.onnx_model_path.is_file()
    assert settings.onnx_model_metadata_path.is_file()
    assert resolve_repo_path("../artifacts/models/onnx/exp04c_yolo26n_opentop_640.onnx") == (
        REPO_ROOT / "artifacts/models/onnx/exp04c_yolo26n_opentop_640.onnx"
    ).resolve()

    monkeypatch.setenv("POSTGRES_PASSWORD", "supersecret-value")
    fresh = Settings(_env_file=None)
    assert "supersecret-value" not in repr(fresh)
    assert fresh.database_url.startswith("postgresql+psycopg://railvision:supersecret-value@")


def test_camel_model_serialization_and_orm_attributes() -> None:
    """CamelModel outputs camelCase JSON while accepting Python snake_case or ORM attributes."""

    class SampleDto(CamelModel):
        inspection_id: str
        file_size_bytes: int
        review_required: bool

    dto = SampleDto(inspection_id="abc", file_size_bytes=42, review_required=False)
    dumped = dto.model_dump(by_alias=True)
    assert dumped == {"inspectionId": "abc", "fileSizeBytes": 42, "reviewRequired": False}

    parsed = SampleDto.model_validate({"inspectionId": "xyz", "fileSizeBytes": 99, "reviewRequired": True})
    assert parsed.inspection_id == "xyz"
    assert parsed.file_size_bytes == 99
    assert parsed.review_required is True

    from_orm = SampleDto.model_validate(
        SimpleNamespace(inspection_id="orm-1", file_size_bytes=123, review_required=True)
    )
    assert from_orm.model_dump(by_alias=True) == {
        "inspectionId": "orm-1",
        "fileSizeBytes": 123,
        "reviewRequired": True,
    }


def test_orm_metadata_matches_flyway_v1_contract() -> None:
    """SQLAlchemy ORM tables, constraints, foreign keys, and indexes mirror V1__initial_schema.sql."""
    expected_tables = {
        "inspections",
        "processing_jobs",
        "model_versions",
        "wagon_results",
        "wagon_observations",
    }
    assert set(Base.metadata.tables.keys()) == expected_tables

    assert [c.name for c in Inspection.__table__.columns] == [
        "id",
        "title",
        "notes",
        "status",
        "original_filename",
        "stored_filename",
        "content_type",
        "file_size_bytes",
        "sha256_checksum",
        "duration_ms",
        "width_px",
        "height_px",
        "fps",
        "codec",
        "latest_job_id",
        "created_at",
        "updated_at",
    ]
    assert [c.name for c in ProcessingJob.__table__.columns] == [
        "id",
        "inspection_id",
        "status",
        "progress_percent",
        "engine_type",
        "is_demo_simulation",
        "error_code",
        "error_message",
        "started_at",
        "finished_at",
        "created_at",
    ]
    assert [c.name for c in ModelVersion.__table__.columns] == [
        "id",
        "task",
        "name",
        "version",
        "runtime",
        "weights_sha256",
        "artifact_ref",
        "is_demo_placeholder",
        "created_at",
    ]
    assert [c.name for c in WagonResult.__table__.columns] == [
        "id",
        "inspection_id",
        "job_id",
        "track_id",
        "sequence_index",
        "physical_id",
        "material_class",
        "material_confidence",
        "material_rejected_as_unknown",
        "vote_distribution_json",
        "loading_level",
        "loading_confidence",
        "loading_spread_json",
        "estimated_fill_percentage",
        "measurement_type",
        "physical_volume_verified",
        "observations_count",
        "consolidation_method",
        "disagreements_json",
        "quality_flags_json",
        "review_required",
        "review_reasons_json",
        "model_versions_json",
        "is_example",
        "created_at",
    ]
    assert [c.name for c in WagonObservation.__table__.columns] == [
        "id",
        "wagon_result_id",
        "frame_index",
        "frame_timestamp_ms",
        "bbox_x",
        "bbox_y",
        "bbox_width",
        "bbox_height",
        "coordinate_system",
        "detected_class",
        "detection_confidence",
        "material_class",
        "material_confidence",
        "loading_level",
        "loading_confidence",
        "is_example",
    ]

    all_index_names = {idx.name for table in Base.metadata.tables.values() for idx in table.indexes}
    assert all_index_names == {
        "idx_inspections_created_at_desc",
        "idx_processing_jobs_inspection_created",
        "idx_processing_jobs_status",
        "uq_processing_jobs_one_active_per_inspection",
        "idx_wagon_results_inspection_job_seq",
        "idx_wagon_observations_result_ts",
    }

    uq_names = {
        c.name
        for table in Base.metadata.tables.values()
        for c in table.constraints
        if isinstance(c, sa.UniqueConstraint)
    }
    assert {"inspections_stored_filename_key", "uq_model_versions_task_name_version_runtime"} <= uq_names


def test_fastapi_health_and_cors() -> None:
    """FastAPI create_app exposes GET /health and configured CORS headers."""
    app = create_app(get_settings(), enable_startup_recovery=False)
    with TestClient(app) as client:
        response = client.get("/health", headers={"Origin": "http://localhost:4200"})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["backend"] == "python-fastapi"
        assert response.headers.get("access-control-allow-origin") == "http://localhost:4200"


@pytest.mark.gpu
def test_cuda_and_pyav_4k_decode_plus_h264_encoder_probe() -> None:
    """CUDA runtime stays intact and PyAV decodes 4K VP9 vid_01 with H.264 encoders available."""
    assert torch.cuda.is_available()
    assert "4060 Ti" in torch.cuda.get_device_name(0)

    assert "libx264" in av.codecs_available
    h264_encoder = av.codec.Codec("libx264", "w")
    assert h264_encoder.name == "libx264"

    vid01_path = (
        REPO_ROOT
        / "data/raw/public-videos/vid_01_langer_unbeladener_g_terzug_von_oben_k_l.webm"
    )
    assert vid01_path.is_file()

    with av.open(str(vid01_path)) as container:
        stream = container.streams.video[0]
        assert (stream.width, stream.height) == (3840, 2160)
        assert stream.codec_context.name == "vp9"
        first_frame = next(container.decode(video=0))
        rgb = first_frame.to_ndarray(format="rgb24")
        assert rgb.shape == (2160, 3840, 3)


@pytest.mark.integration
def test_postgresql_readonly_public_and_isolated_alembic_roundtrip() -> None:
    """Inspect public schema read-only without changes, then test Alembic upgrade/downgrade in temp schema."""
    settings = get_settings()
    engine = sa.create_engine(settings.database_url, poolclass=sa.pool.NullPool, future=True)
    temp_schema = f"railvision_m0_alembic_test_{uuid4().hex[:8]}"

    with engine.connect() as conn:
        public_tables_before = sorted(sa.inspect(conn).get_table_names(schema="public"))

    cfg = _alembic_config(schema=temp_schema, database_url=settings.database_url)

    try:
        with engine.begin() as conn:
            conn.exec_driver_sql(f'CREATE SCHEMA "{temp_schema}"')

        command.upgrade(cfg, "head")

        with engine.connect() as conn:
            inspector = sa.inspect(conn)
            created_tables = set(inspector.get_table_names(schema=temp_schema))
            assert created_tables == {
                "alembic_version",
                "inspections",
                "processing_jobs",
                "model_versions",
                "wagon_results",
                "wagon_observations",
            }
            for table_name, orm_table in Base.metadata.tables.items():
                migrated_cols = {col["name"] for col in inspector.get_columns(table_name, schema=temp_schema)}
                expected_cols = {col.name for col in orm_table.columns}
                assert migrated_cols == expected_cols

            migrated_indexes = {
                idx["name"]
                for table_name in Base.metadata.tables
                for idx in inspector.get_indexes(table_name, schema=temp_schema)
            }
            assert {
                "idx_inspections_created_at_desc",
                "idx_processing_jobs_inspection_created",
                "idx_processing_jobs_status",
                "uq_processing_jobs_one_active_per_inspection",
                "idx_wagon_results_inspection_job_seq",
                "idx_wagon_observations_result_ts",
            } <= migrated_indexes

        # Verify ORM round-trip write/read inside the isolated temp schema only.
        isolated_engine = sa.create_engine(
            settings.database_url,
            poolclass=sa.pool.NullPool,
            connect_args={"options": f"-c search_path={temp_schema}"},
            future=True,
        )
        now = datetime.now(timezone.utc)
        with isolated_engine.begin() as iso_conn:
            iso_conn.execute(
                sa.insert(Inspection).values(
                    id=uuid4(),
                    title="M0 Smoke Check",
                    notes=None,
                    status="UPLOADED",
                    original_filename="sample.mp4",
                    stored_filename=f"{uuid4()}.mp4",
                    content_type="video/mp4",
                    file_size_bytes=1024,
                    sha256_checksum="0" * 64,
                    created_at=now,
                    updated_at=now,
                )
            )
            count = iso_conn.execute(sa.select(sa.func.count()).select_from(Inspection)).scalar_one()
            assert count == 1
        isolated_engine.dispose()

        command.downgrade(cfg, "base")

        with engine.connect() as conn:
            remaining = set(sa.inspect(conn).get_table_names(schema=temp_schema))
            assert remaining == {"alembic_version"}

            public_tables_after = sorted(sa.inspect(conn).get_table_names(schema="public"))
            assert public_tables_after == public_tables_before
    finally:
        with engine.begin() as conn:
            conn.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{temp_schema}" CASCADE')
        engine.dispose()
