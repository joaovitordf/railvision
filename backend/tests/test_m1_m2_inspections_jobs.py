"""Stages M1 & M2 integration tests: validation, uploads, Range streaming, async jobs, recovery."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
from uuid import uuid4

from alembic import command
from alembic.config import Config as AlembicConfig
import av
from fastapi.testclient import TestClient
import pytest
import sqlalchemy as sa

from app.core.config import Settings, get_settings
from app.core.paths import BACKEND_ROOT, REPO_ROOT
from app.db.models import Inspection, ProcessingJob
from app.db.session import create_db_engine, get_session_factory
from app.main import create_app
from app.modules.jobs.engine import DemoFixtureInferenceEngine
from app.modules.jobs.recovery import recover_interrupted_jobs


def _make_settings(tmp_path: Path, *, max_mb: int = 10, demo_delay_ms: int = 80) -> Settings:
    base = get_settings()
    return base.model_copy(
        update={
            "storage_root_raw": str(tmp_path / "storage"),
            "onnx_runs_storage_path_raw": str(tmp_path / "lab-runs"),
            "upload_max_size_mb": max_mb,
            "processing_demo_delay_ms": demo_delay_ms,
        }
    )


def _alembic_cfg(schema: str, database_url: str) -> AlembicConfig:
    cfg = AlembicConfig(str(BACKEND_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    cfg.attributes["schema"] = schema
    cfg.attributes["database_url"] = database_url
    return cfg


def _synthetic_mp4(size: int = 64) -> bytes:
    header = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    if size <= len(header):
        return header[:size]
    return header + b"A" * (size - len(header))


@pytest.fixture()
def isolated_db(tmp_path: Path):
    """Provision an isolated temporary PostgreSQL schema migrated with Alembic, leaving public untouched."""
    settings = _make_settings(tmp_path, demo_delay_ms=150)
    admin_engine = create_db_engine(settings.database_url)
    temp_schema = f"railvision_m12_test_{uuid4().hex[:8]}"

    with admin_engine.connect() as conn:
        public_tables_before = sorted(sa.inspect(conn).get_table_names(schema="public"))

    try:
        with admin_engine.begin() as conn:
            conn.exec_driver_sql(f'CREATE SCHEMA "{temp_schema}"')
        command.upgrade(_alembic_cfg(temp_schema, settings.database_url), "head")

        iso_engine = create_db_engine(settings.database_url, schema=temp_schema)
        iso_factory = get_session_factory(settings.database_url, schema=temp_schema)
        yield settings, temp_schema, iso_engine, iso_factory
    finally:
        if "iso_engine" in locals():
            iso_engine.dispose()
        with admin_engine.begin() as conn:
            conn.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{temp_schema}" CASCADE')
        with admin_engine.connect() as conn:
            public_tables_after = sorted(sa.inspect(conn).get_table_names(schema="public"))
            assert public_tables_after == public_tables_before
        admin_engine.dispose()


def test_upload_validation_and_problem_detail_responses(isolated_db) -> None:
    """Reject empty, oversized, bad extension, traversal, and invalid container magic bytes with ProblemDetail."""
    settings, _, _, iso_factory = isolated_db
    small_settings = settings.model_copy(update={"upload_max_size_mb": 1})
    app = create_app(small_settings, session_factory=iso_factory)

    with TestClient(app) as client:
        empty_resp = client.post(
            "/api/v1/inspections",
            files={"file": ("empty.mp4", b"", "video/mp4")},
        )
        assert empty_resp.status_code == 400
        assert empty_resp.headers["content-type"].startswith("application/problem+json")
        assert empty_resp.json()["errorCode"] == "EMPTY_FILE"

        bad_ext = client.post(
            "/api/v1/inspections",
            files={"file": ("script.exe", b"MZ" + b"\x00" * 32, "application/octet-stream")},
        )
        assert bad_ext.status_code == 400
        assert bad_ext.json()["errorCode"] == "UNSUPPORTED_EXTENSION"

        bad_sig = client.post(
            "/api/v1/inspections",
            files={"file": ("fake.mp4", b"NOT_AN_MP4_CONTAINER_HEADER", "video/mp4")},
        )
        assert bad_sig.status_code == 400
        assert bad_sig.json()["errorCode"] == "INVALID_VIDEO_SIGNATURE"

        too_large = client.post(
            "/api/v1/inspections",
            files={"file": ("huge.mp4", _synthetic_mp4(1024 * 1024 + 512), "video/mp4")},
        )
        assert too_large.status_code == 413
        assert too_large.headers["content-type"].startswith("application/problem+json")


def test_full_inspection_media_range_and_async_demo_job_workflow(isolated_db) -> None:
    """Verify upload -> Range streaming -> injected fixture job mapping (incl. empty-wagon rule)."""
    settings, _, _, iso_factory = isolated_db
    app = create_app(
        settings,
        session_factory=iso_factory,
        inference_engine=DemoFixtureInferenceEngine(demo_delay_ms=settings.processing_demo_delay_ms),
    )
    payload_bytes = _synthetic_mp4(128)

    with TestClient(app) as client:
        # Health checks
        assert client.get("/health").status_code == 200
        act = client.get("/actuator/health")
        assert act.status_code == 200
        assert act.json()["status"] == "UP"
        assert act.json()["components"]["railvision"]["details"]["backend"] == "python-fastapi"

        # Upload inspection
        create_resp = client.post(
            "/api/v1/inspections",
            data={"title": "Composição Norte", "notes": "Teste M1-M2"},
            files={"file": ("train_sample.mp4", payload_bytes, "video/mp4")},
        )
        assert create_resp.status_code == 201
        created = create_resp.json()
        insp_id = created["id"]
        assert create_resp.headers["location"].endswith(f"/api/v1/inspections/{insp_id}")
        assert created["title"] == "Composição Norte"
        assert created["status"] == "UPLOADED"
        assert created["fileSizeBytes"] == len(payload_bytes)

        # List and detail
        listed = client.get("/api/v1/inspections")
        assert listed.status_code == 200
        assert [row["id"] for row in listed.json()] == [insp_id]

        detail_before = client.get(f"/api/v1/inspections/{insp_id}").json()
        assert detail_before["wagonCount"] == 0
        assert detail_before["latestJob"] is None

        # Media full 200, partial 206, and invalid Range 416
        full_media = client.get(f"/api/v1/inspections/{insp_id}/media")
        assert full_media.status_code == 200
        assert full_media.headers["accept-ranges"] == "bytes"
        assert full_media.content == payload_bytes

        range_resp = client.get(f"/api/v1/inspections/{insp_id}/media", headers={"Range": "bytes=4-15"})
        assert range_resp.status_code == 206
        assert range_resp.headers["content-range"] == f"bytes 4-15/{len(payload_bytes)}"
        assert range_resp.content == payload_bytes[4:16]

        bad_range = client.get(f"/api/v1/inspections/{insp_id}/media", headers={"Range": "bytes=9999-"})
        assert bad_range.status_code == 416
        assert bad_range.headers["content-range"] == f"bytes */{len(payload_bytes)}"

        # Enqueue async job and check duplicate conflict
        proc_resp = client.post(f"/api/v1/inspections/{insp_id}/process")
        assert proc_resp.status_code == 202
        job = proc_resp.json()
        job_id = job["id"]
        assert job["status"] == "QUEUED"
        assert job["engineType"] == "DEMO_FIXTURE_ENGINE"
        assert job["isDemoSimulation"] is True

        conflict_resp = client.post(f"/api/v1/inspections/{insp_id}/process")
        assert conflict_resp.status_code == 409

        # Poll job completion
        deadline = time.monotonic() + 10.0
        final_job = job
        while time.monotonic() < deadline and final_job["status"] in {"QUEUED", "RUNNING"}:
            time.sleep(0.05)
            polled = client.get(f"/api/v1/jobs/{job_id}")
            assert polled.status_code == 200
            final_job = polled.json()

        assert final_job["status"] == "COMPLETED"
        assert final_job["progressPercent"] == 100

        detail_after = client.get(f"/api/v1/inspections/{insp_id}").json()
        assert detail_after["status"] == "COMPLETED"
        assert detail_after["wagonCount"] == 3
        assert detail_after["reviewRequiredCount"] == 1

        # Verify wagons, empty wagon rule, observations, and models
        wagons = client.get(f"/api/v1/inspections/{insp_id}/wagons").json()
        assert len(wagons) == 3
        assert [w["wagonIdentity"]["trackId"] for w in wagons] == [
            "example-track-001",
            "example-track-002",
            "example-track-003",
        ]
        empty_wagon = wagons[2]
        assert empty_wagon["loadingEstimate"]["level"] == "empty"
        assert empty_wagon["material"] is None

        wagon_detail = client.get(f"/api/v1/inspections/{insp_id}/wagons/{wagons[0]['id']}").json()
        assert wagon_detail["material"]["materialClass"] == "iron_ore"

        obs_list = client.get(f"/api/v1/inspections/{insp_id}/wagons/{wagons[0]['id']}/observations").json()
        assert len(obs_list) == 2
        assert obs_list[0]["boundingBox"]["coordinateSystem"] == "normalized"

        models = client.get("/api/v1/models?includePlaceholders=true").json()
        assert any(m["isDemoPlaceholder"] is True for m in models)


def _export_short_vid15_mp4(out_path: Path, num_frames: int = 16) -> Path:
    src = REPO_ROOT / "data/raw/public-videos/vid_15_sishen_saldanha_railway_line.ogv"
    with av.open(str(src)) as inp, av.open(str(out_path), mode="w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=5)
        stream.width = 1920
        stream.height = 1080
        stream.pix_fmt = "yuv420p"
        written = 0
        for idx, frame in enumerate(inp.decode(video=0)):
            if idx % 5 != 0:
                continue
            bgr = frame.to_ndarray(format="bgr24")
            nf = av.VideoFrame.from_ndarray(bgr, format="bgr24")
            nf.pts = written
            for pkt in stream.encode(nf):
                out.mux(pkt)
            written += 1
            if written >= num_frames:
                break
        for pkt in stream.encode(None):
            out.mux(pkt)
    return out_path


def test_real_vision_inspection_workflow_gpu_and_bytetrack(isolated_db, tmp_path: Path) -> None:
    """Run default VisionInspectionInferenceEngine on MP4 clip verifying real GPU detections and not_evaluated cargo."""
    settings, _, _, iso_factory = isolated_db
    capped_settings = settings.model_copy(update={"max_video_frames_per_ui_run": 16})
    clip_path = _export_short_vid15_mp4(tmp_path / "vid15_sample.mp4", num_frames=16)

    app = create_app(capped_settings, session_factory=iso_factory)
    with TestClient(app) as client:
        # Startup registers real active models and excludes demo placeholders by default
        models_resp = client.get("/api/v1/models")
        assert models_resp.status_code == 200
        models = models_resp.json()
        assert len(models) >= 2
        assert all(m["isDemoPlaceholder"] is False for m in models)
        assert {m["task"] for m in models} >= {"detector", "tracker"}

        create_resp = client.post(
            "/api/v1/inspections",
            data={"title": "Trecho Real Sishen MP4"},
            files={"file": (clip_path.name, clip_path.read_bytes(), "video/mp4")},
        )
        assert create_resp.status_code == 201
        insp_id = create_resp.json()["id"]

        proc_resp = client.post(f"/api/v1/inspections/{insp_id}/process")
        assert proc_resp.status_code == 202
        job = proc_resp.json()
        assert job["engineType"] == "VISION_PYTORCH_CUDA"
        assert job["isDemoSimulation"] is False

        deadline = time.monotonic() + 30.0
        final_job = job
        while time.monotonic() < deadline and final_job["status"] in {"QUEUED", "RUNNING"}:
            time.sleep(0.1)
            final_job = client.get(f"/api/v1/jobs/{job['id']}").json()

        assert final_job["status"] == "COMPLETED", final_job.get("errorMessage")

        detail = client.get(f"/api/v1/inspections/{insp_id}").json()
        assert detail["status"] == "COMPLETED"
        assert detail["widthPx"] == 1920
        assert detail["heightPx"] == 1080
        assert detail["codec"] == "h264"
        assert detail["wagonCount"] >= 1

        wagons = client.get(f"/api/v1/inspections/{insp_id}/wagons").json()
        assert len(wagons) == detail["wagonCount"]
        first = wagons[0]
        assert first["isExample"] is False
        assert first["material"]["materialClass"] == "not_evaluated"
        assert first["loadingEstimate"]["level"] == "not_evaluated"
        assert first["consolidation"]["method"] == "bytetrack_detection_only"

        obs = client.get(f"/api/v1/inspections/{insp_id}/wagons/{first['id']}/observations").json()
        assert len(obs) >= 1
        assert obs[0]["isExample"] is False
        assert obs[0]["detectionConfidence"] > 0.0
        assert obs[0]["detectedClass"] == "open_top_freight_wagon"


def test_stale_job_recovery_marks_interrupted_jobs_and_inspections_failed(isolated_db) -> None:
    """Verify recover_interrupted_jobs marks leftover QUEUED/RUNNING rows as FAILED (PROCESS_INTERRUPTED)."""
    _, _, _, iso_factory = isolated_db
    now = datetime.now(timezone.utc)
    insp_id = uuid4()
    job_id = uuid4()

    with iso_factory() as db:
        db.add(
            Inspection(
                id=insp_id,
                title="Interrupted Sample",
                notes=None,
                status="PROCESSING",
                original_filename="stale.mp4",
                stored_filename=f"{uuid4()}.mp4",
                content_type="video/mp4",
                file_size_bytes=64,
                sha256_checksum="a" * 64,
                latest_job_id=job_id,
                created_at=now,
                updated_at=now,
            )
        )
        db.add(
            ProcessingJob(
                id=job_id,
                inspection_id=insp_id,
                status="RUNNING",
                progress_percent=15,
                engine_type="DEMO_FIXTURE_ENGINE",
                is_demo_simulation=True,
                started_at=now,
                created_at=now,
            )
        )
        db.commit()

    recovered = recover_interrupted_jobs(iso_factory)
    assert recovered == 1

    with iso_factory() as db:
        job = db.get(ProcessingJob, job_id)
        insp = db.get(Inspection, insp_id)
        assert job is not None and insp is not None
        assert job.status == "FAILED"
        assert job.error_code == "PROCESS_INTERRUPTED"
        assert insp.status == "FAILED"
