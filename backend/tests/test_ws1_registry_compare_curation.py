"""WS1 backend tests: Model Registry, CUDA 1-model LRU, /api/v1/vision-lab compare & HITL curation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.core.config import get_settings
from app.core.paths import REPO_ROOT
from app.main import create_app
from app.modules.vision.registry import ACTIVE_DEFAULT_MODEL_ID, ModelRegistryService
from app.modules.vision.runtimes import VisionRuntimeManager


@pytest.mark.unit
def test_model_registry_lists_open_top_checkpoints_only(tmp_path: Path) -> None:
    """ModelRegistryService registers exp04c active + exp04a/exp04b archived and excludes TWIST 2-class runs."""
    reg_path = tmp_path / "registry.json"
    settings = get_settings().model_copy(update={"model_registry_path_raw": str(reg_path)})
    registry = ModelRegistryService(settings)

    checkpoints = registry.list_checkpoints()
    ids = [cp.id for cp in checkpoints]
    assert ids[0] == ACTIVE_DEFAULT_MODEL_ID
    assert registry.get_active_checkpoint().status == "active"
    assert not any(cp_id.startswith(("exp01_", "exp02_", "exp03_")) for cp_id in ids)
    assert any("exp04a" in cp_id for cp_id in ids)
    assert any("exp04b" in cp_id for cp_id in ids)
    assert reg_path.is_file()


@pytest.mark.unit
def test_cuda_lru_cache_evicts_previous_model_on_switch(tmp_path: Path) -> None:
    """Strict 1-model CUDA LRU keeps at most one checkpoint resident on GPU at any time."""
    reg_path = tmp_path / "registry.json"
    settings = get_settings().model_copy(update={"model_registry_path_raw": str(reg_path)})
    mgr = VisionRuntimeManager(settings)

    class _DummyTensor:
        def detach(self) -> "_DummyTensor":
            return self

        def cpu(self) -> "_DummyTensor":
            return self

        def numpy(self) -> np.ndarray:
            return np.zeros((1, 5, 1), dtype=np.float32)

    class _DummyModel:
        def eval(self) -> "_DummyModel":
            return self

        def to(self, _device: str) -> "_DummyModel":
            return self

        def __call__(self, _x: object) -> _DummyTensor:
            return _DummyTensor()

    class _DummyYolo:
        def __init__(self, _weights: str) -> None:
            self.model = _DummyModel()

    with (
        patch("app.modules.vision.runtimes.YOLO", _DummyYolo),
        patch("app.modules.vision.runtimes.torch.zeros", return_value="dummy"),
        patch("app.modules.vision.runtimes.torch.from_numpy", return_value=_DummyModel()),
        patch("app.modules.vision.runtimes.torch.cuda.is_available", return_value=True),
        patch("app.modules.vision.runtimes.torch.cuda.get_device_name", return_value="Mock RTX 4060 Ti"),
        patch("app.modules.vision.runtimes.torch.cuda.synchronize"),
        patch("app.modules.vision.runtimes.torch.cuda.empty_cache") as empty_cache_mock,
    ):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        mgr.detect_bgr(
            frame,
            conf_threshold=0.25,
            iou_threshold=0.45,
            runtime="pytorch_cuda",
            model_id="exp04c_yolo26n_opentop_640",
        )
        assert len(mgr._pt_cuda_models) == 1
        assert empty_cache_mock.call_count == 0

        mgr.detect_bgr(
            frame,
            conf_threshold=0.25,
            iou_threshold=0.45,
            runtime="pytorch_cuda",
            model_id="exp04b_yolo26n_opentop_from_twist_exp02",
        )
        assert len(mgr._pt_cuda_models) == 1
        assert empty_cache_mock.call_count == 1


@pytest.mark.unit
def test_vision_lab_checkpoints_compare_and_curate_image_and_video(tmp_path: Path) -> None:
    """Exercise /api/v1/models/checkpoints, /api/v1/vision-lab/* alias, modelId, compare, and clean curation."""
    settings = get_settings().model_copy(
        update={
            "storage_root_raw": str(tmp_path / "storage"),
            "onnx_runs_storage_path_raw": str(tmp_path / "lab-runs"),
            "model_registry_path_raw": str(tmp_path / "registry.json"),
            "curation_inbox_path_raw": str(tmp_path / "curation-inbox"),
            "max_video_frames_per_ui_run": 10,
            "default_runtime": "pytorch_cpu",
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
        cp_resp = client.get("/api/v1/models/checkpoints")
        assert cp_resp.status_code == 200
        checkpoints = cp_resp.json()
        assert checkpoints[0]["id"] == ACTIVE_DEFAULT_MODEL_ID
        assert checkpoints[0]["status"] == "active"

        vl_status = client.get("/api/v1/vision-lab/status")
        ox_status = client.get("/api/v1/onnx-spike/status")
        assert vl_status.status_code == 200 and ox_status.status_code == 200
        assert vl_status.json()["activeModelId"] == ACTIVE_DEFAULT_MODEL_ID

        run_a_resp = client.post(
            "/api/v1/vision-lab/analyze-image?conf=0.25&iou=0.45&runtime=pytorch_cpu&modelId=exp04c_yolo26n_opentop_640",
            files={"file": (val_img.name, val_img.read_bytes(), "image/jpeg")},
        )
        assert run_a_resp.status_code == 200
        run_a = run_a_resp.json()
        assert run_a["modelConfig"]["modelId"] == "exp04c_yolo26n_opentop_640"

        run_b_resp = client.post(
            "/api/v1/vision-lab/analyze-image?conf=0.25&iou=0.45&runtime=pytorch_cpu&modelId=exp04a_yolo26n_opentop_from_coco",
            files={"file": (val_img.name, val_img.read_bytes(), "image/jpeg")},
        )
        assert run_b_resp.status_code == 200
        run_b = run_b_resp.json()
        assert run_b["modelConfig"]["modelId"] == "exp04a_yolo26n_opentop_from_coco"

        cmp_resp = client.get(f"/api/v1/vision-lab/compare?runA={run_a['runId']}&runB={run_b['runId']}")
        assert cmp_resp.status_code == 200
        cmp_data = cmp_resp.json()
        assert cmp_data["runA"]["modelId"] == "exp04c_yolo26n_opentop_640"
        assert cmp_data["runB"]["modelId"] == "exp04a_yolo26n_opentop_from_coco"
        assert cmp_data["comparable"] is True
        assert cmp_data["comparabilityWarnings"] == []
        assert cmp_data["deltaDetections"] == cmp_data["runB"]["totalDetections"] - cmp_data["runA"]["totalDetections"]
        assert run_a["files"]["annotatedUrl"].startswith("/api/v1/vision-lab/runs/")

        # Different thresholds trigger explicit comparability warning
        run_c_resp = client.post(
            "/api/v1/vision-lab/analyze-image?conf=0.55&iou=0.45&runtime=pytorch_cpu&modelId=exp04c_yolo26n_opentop_640",
            files={"file": (val_img.name, val_img.read_bytes(), "image/jpeg")},
        )
        assert run_c_resp.status_code == 200
        cmp_warn = client.get(f"/api/v1/vision-lab/compare?runA={run_a['runId']}&runB={run_c_resp.json()['runId']}")
        assert cmp_warn.status_code == 200
        assert cmp_warn.json()["comparable"] is False
        assert any("Limiares" in w for w in cmp_warn.json()["comparabilityWarnings"])

        # Curate image frame -> HTTP 201 + clean JPEG + YOLO .txt + SHA provenance + humanReviewed=False
        cur_img_resp = client.post(
            f"/api/v1/vision-lab/runs/{run_a['runId']}/curate-frame",
            json={"frameIndex": 0, "reason": "false_negative_edge", "notes": "Verificar vagao parcialmente oculto"},
        )
        assert cur_img_resp.status_code == 201
        cur_img = cur_img_resp.json()
        assert cur_img["humanReviewed"] is False
        assert cur_img["autoTrainEligible"] is False
        assert cur_img["detectionCount"] == run_a["detectionCount"]
        assert len(cur_img["sourceSha256"]) == 64
        assert cur_img["reservedBenchmarkWarning"] is False

        # Re-curating same frame uses collision-safe filename
        cur_img_dup = client.post(
            f"/api/v1/vision-lab/runs/{run_a['runId']}/curate-frame",
            json={"frameIndex": 0, "reason": "bad_bbox"},
        )
        assert cur_img_dup.status_code == 201
        assert cur_img_dup.json()["imagePath"] != cur_img["imagePath"]

        saved_meta = json.loads(Path(cur_img["metadataPath"]).read_text(encoding="utf-8"))
        assert saved_meta["humanReviewed"] is False
        assert saved_meta["sourceSha256"] == cur_img["sourceSha256"]
        assert saved_meta["reason"] == "false_negative_edge"
        saved_bgr = cv2.imread(cur_img["imagePath"])
        orig_bgr = cv2.imread(str(val_img))
        assert saved_bgr is not None and orig_bgr is not None
        assert saved_bgr.shape == orig_bgr.shape

        # Video curation decodes exact clean frame from original video
        vid_sub = client.post(
            "/api/v1/vision-lab/analyze-video?conf=0.25&iou=0.45&sampleFps=2&runtime=onnx_cpu",
            files={"file": (vid15.name, vid15.read_bytes(), "video/ogg")},
        )
        assert vid_sub.status_code == 202
        vid_id = vid_sub.json()["runId"]
        deadline = time_monotonic_deadline(30.0)
        import time

        vid_payload = vid_sub.json()
        while time.monotonic() < deadline and vid_payload["status"] in {"QUEUED", "RUNNING"}:
            time.sleep(0.1)
            vid_payload = client.get(f"/api/v1/vision-lab/runs/{vid_id}").json()
        assert vid_payload["status"] == "COMPLETED"
        first_idx = vid_payload["frameSummaries"][0]["frameIndex"]

        cur_vid_resp = client.post(
            f"/api/v1/vision-lab/runs/{vid_id}/curate-frame",
            json={"frameIndex": first_idx, "reason": "hard_sample_review"},
        )
        assert cur_vid_resp.status_code == 201
        cur_vid = cur_vid_resp.json()
        assert cur_vid["frameIndex"] == first_idx
        assert Path(cur_vid["imagePath"]).is_file()
        assert Path(cur_vid["labelPath"]).is_file()


@pytest.mark.unit
def test_cuda_oom_clears_cache_and_raises_explicit_runtime_unavailable(tmp_path: Path) -> None:
    """CUDA OutOfMemoryError clears GPU cache and raises RuntimeUnavailableException without silent CPU fallback."""
    import torch
    from app.core.exceptions import RuntimeUnavailableException

    settings = get_settings().model_copy(update={"model_registry_path_raw": str(tmp_path / "registry.json")})
    mgr = VisionRuntimeManager(settings)

    class _OomModel:
        def eval(self):
            return self

        def to(self, _dev: str):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")

    class _OomYolo:
        def __init__(self, _w: str) -> None:
            self.model = _OomModel()

    with (
        patch("app.modules.vision.runtimes.YOLO", _OomYolo),
        patch("app.modules.vision.runtimes.torch.cuda.is_available", return_value=True),
        patch("app.modules.vision.runtimes.torch.cuda.empty_cache") as empty_cache_mock,
    ):
        with pytest.raises(RuntimeUnavailableException, match="CUDA OutOfMemoryError"):
            mgr.detect_bgr(
                np.zeros((64, 64, 3), dtype=np.uint8),
                conf_threshold=0.25,
                iou_threshold=0.45,
                runtime="pytorch_cuda",
                allow_cpu_fallback=True,
                model_id="exp04c_yolo26n_opentop_640",
            )
        assert len(mgr._pt_cuda_models) == 0
        assert empty_cache_mock.call_count >= 1


@pytest.mark.gpu
def test_real_gpu_lru_switch_between_registered_checkpoints(tmp_path: Path) -> None:
    """Real RTX 4060 Ti model switch between exp04c and exp04a keeps CUDA cache size <= 1."""
    settings = get_settings().model_copy(update={"model_registry_path_raw": str(tmp_path / "registry.json")})
    mgr = VisionRuntimeManager(settings)
    val_img = (
        REPO_ROOT
        / "artifacts/runs/phase4-onnx-spike/parity_frames/val__camp_sishen_saldanha__vid_15__f000075.jpg"
    )
    bgr = cv2.imread(str(val_img))
    assert bgr is not None

    r1 = mgr.detect_bgr(bgr, conf_threshold=0.25, iou_threshold=0.45, runtime="pytorch_cuda", model_id="exp04c_yolo26n_opentop_640")
    assert r1.device_type == "GPU"
    assert len(mgr._pt_cuda_models) == 1

    r2 = mgr.detect_bgr(bgr, conf_threshold=0.25, iou_threshold=0.45, runtime="pytorch_cuda", model_id="exp04a_yolo26n_opentop_from_coco")
    assert r2.device_type == "GPU"
    assert len(mgr._pt_cuda_models) == 1
    assert mgr.model_info_for_runtime("pytorch_cuda", model_id="exp04c_yolo26n_opentop_640")[1] != mgr.model_info_for_runtime("pytorch_cuda", model_id="exp04a_yolo26n_opentop_from_coco")[1]


@pytest.mark.unit
def test_iter_decoded_video_frames_ignores_trailing_zero_byte_packets() -> None:
    """iter_decoded_video_frames ignores 0-byte timestamped EOF packets and decodes full .ogv streams cleanly."""
    import av
    from app.modules.vision.video_pipeline import iter_decoded_video_frames

    class _FakePacket:
        def __init__(self, *, size: int, dts: int | None, frames=None, exc=None) -> None:
            self.size = size
            self.dts = dts
            self._frames = frames or []
            self._exc = exc
            self.decode_called = False

        def decode(self):
            self.decode_called = True
            if self._exc is not None:
                raise self._exc
            return list(self._frames)

    class _FakeContainer:
        def __init__(self, packets) -> None:
            self._packets = packets

        def demux(self, video: int = 0):
            assert video == 0
            return iter(self._packets)

    empty_ts_pkt = _FakePacket(size=0, dts=787, exc=av.error.ArgumentError(22, "einval"))
    trailing_err_pkt = _FakePacket(size=4, dts=788, exc=av.error.InvalidDataError(1, "trailing"))
    flush_pkt = _FakePacket(size=0, dts=None, frames=["flushed_frame"])
    fake = _FakeContainer([
        _FakePacket(size=100, dts=0, frames=["frame_0"]),
        empty_ts_pkt,
        trailing_err_pkt,
        flush_pkt,
    ])
    decoded = list(iter_decoded_video_frames(fake))
    assert decoded == ["frame_0", "flushed_frame"]
    assert empty_ts_pkt.decode_called is False
    assert flush_pkt.decode_called is True

    vid15 = REPO_ROOT / "data/raw/public-videos/vid_15_sishen_saldanha_railway_line.ogv"
    if vid15.is_file():
        with av.open(str(vid15)) as real_container:
            frames = list(iter_decoded_video_frames(real_container))
        assert len(frames) >= 700


def time_monotonic_deadline(seconds: float) -> float:
    import time

    return time.monotonic() + seconds
