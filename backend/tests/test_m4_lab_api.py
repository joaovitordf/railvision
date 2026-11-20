"""Stage M4 HTTP end-to-end tests for /api/v1/onnx-spike viewer endpoints."""

from __future__ import annotations

from pathlib import Path
import time

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.core.paths import REPO_ROOT
from app.main import create_app


def test_lab_status_analyze_image_video_bytetrack_and_history(tmp_path: Path) -> None:
    """Exercise /api/v1/onnx-spike status, CUDA vs ONNX CPU image inference, tracked video, and Range streaming."""
    settings = get_settings().model_copy(
        update={
            "storage_root_raw": str(tmp_path / "app-storage"),
            "onnx_runs_storage_path_raw": str(tmp_path / "lab-runs"),
            "max_video_frames_per_ui_run": 15,
        }
    )
    app = create_app(settings, enable_startup_recovery=False)

    val_img = (
        REPO_ROOT
        / "artifacts/runs/phase4-onnx-spike/parity_frames/val__camp_sishen_saldanha__vid_15__f000075.jpg"
    )
    vid15 = REPO_ROOT / "data/raw/public-videos/vid_15_sishen_saldanha_railway_line.ogv"
    assert val_img.is_file() and vid15.is_file()

    with TestClient(app) as client:
        st_resp = client.get("/api/v1/onnx-spike/status")
        assert st_resp.status_code == 200
        st = st_resp.json()
        assert st["available"] is True
        assert st["backendEngine"] == "python-fastapi"
        assert st["defaultRuntime"] == "pytorch_cuda"
        assert st["deviceType"] == "GPU"
        assert st["videoEncoder"] == "libx264"
        assert st["defaultOutputResolution"] == "1080p"
        assert st["trackingAlgorithms"] == ["none", "bytetrack"]
        assert [r["id"] for r in st["availableRuntimes"]] == ["pytorch_cuda", "pytorch_cpu", "onnx_cpu"]

        # PyTorch CUDA image inference
        cuda_img_resp = client.post(
            "/api/v1/onnx-spike/analyze-image?conf=0.25&iou=0.45&runtime=pytorch_cuda",
            files={"file": (val_img.name, val_img.read_bytes(), "image/jpeg")},
        )
        assert cuda_img_resp.status_code == 200
        cuda_img = cuda_img_resp.json()
        assert cuda_img["detectionCount"] == 4
        assert cuda_img["modelConfig"]["deviceType"] == "GPU"
        assert cuda_img["modelConfig"]["runtimeId"] == "pytorch_cuda"

        # ONNX Runtime CPU image inference
        onnx_img_resp = client.post(
            "/api/v1/onnx-spike/analyze-image?conf=0.25&iou=0.45&runtime=onnx_cpu",
            files={"file": (val_img.name, val_img.read_bytes(), "image/jpeg")},
        )
        assert onnx_img_resp.status_code == 200
        onnx_img = onnx_img_resp.json()
        assert onnx_img["detectionCount"] == 4
        assert onnx_img["modelConfig"]["deviceType"] == "CPU"
        assert onnx_img["modelConfig"]["activeExecutionProvider"] == "CPUExecutionProvider"

        # Annotated PNG stream & download
        ann_img = client.get(cuda_img["files"]["annotatedDownloadUrl"])
        assert ann_img.status_code == 200
        assert ann_img.headers["content-type"] == "image/png"
        assert "attachment" in ann_img.headers["content-disposition"]

        # Video analysis with PyTorch CUDA + 1080p libx264 + ByteTrack enabled
        vid_submit = client.post(
            "/api/v1/onnx-spike/analyze-video"
            "?conf=0.25&iou=0.45&sampleFps=5&runtime=pytorch_cuda&outputResolution=1080p&enableTracking=true",
            files={"file": (vid15.name, vid15.read_bytes(), "video/ogg")},
        )
        assert vid_submit.status_code == 202
        run_id = vid_submit.json()["runId"]

        deadline = time.monotonic() + 30.0
        vid_run = vid_submit.json()
        while time.monotonic() < deadline and vid_run["status"] in {"QUEUED", "RUNNING"}:
            time.sleep(0.1)
            poll = client.get(f"/api/v1/onnx-spike/runs/{run_id}")
            assert poll.status_code == 200
            vid_run = poll.json()

        assert vid_run["status"] == "COMPLETED", vid_run.get("errorMessage")
        assert vid_run["outputCodec"] == "h264"
        assert vid_run["outputEncoder"] == "libx264"
        assert vid_run["outputWidth"] == 1920
        assert vid_run["outputHeight"] == 1080
        assert vid_run["browserPlayable"] is True
        assert vid_run["modelConfig"]["trackingIntegrated"] is True
        assert vid_run["modelConfig"]["trackerAlgorithm"] == "bytetrack"
        assert vid_run["totalDetections"] > 0
        assert vid_run["uniqueTrackCount"] >= 1

        # Annotated video full + partial Range stream
        ann_vid = client.get(f"/api/v1/onnx-spike/runs/{run_id}/files/annotated", headers={"Range": "bytes=0-1023"})
        assert ann_vid.status_code == 206
        assert ann_vid.headers["content-type"] == "video/mp4"
        assert ann_vid.headers["content-range"].startswith("bytes 0-1023/")

        runs_resp = client.get("/api/v1/onnx-spike/runs")
        assert runs_resp.status_code == 200
        run_ids = [r["runId"] for r in runs_resp.json()]
        assert run_id in run_ids
        assert str(cuda_img["runId"]) in run_ids


def test_lab_lists_preserved_java_runs_cleanly() -> None:
    """Ensure LabService reads existing data/processed/onnx-spike-runs directory without failing."""
    app = create_app(get_settings(), enable_startup_recovery=False)
    with TestClient(app) as client:
        resp = client.get("/api/v1/onnx-spike/runs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
