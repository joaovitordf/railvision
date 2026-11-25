"""Tests for managed sequential YOLO detector training, lineage tracking, GPU locking, and registry promotion."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from uuid import UUID

from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import create_app
from app.modules.training.schemas import RegisterTrainingCheckpointRequest, StartTrainingRequest
from app.modules.training.service import TrainingService, parse_results_csv
from app.modules.vision.registry import ModelRegistryService
from app.modules.vision.runtimes import VisionRuntimeManager


def _create_yolo_dataset(root: Path, name: str, *, val_campaigns: list[str]) -> Path:
    ds = root / name
    for sp in ("train", "val", "test"):
        (ds / sp / "images").mkdir(parents=True, exist_ok=True)
        (ds / sp / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "train" / "images" / "train__camp_a__img1.jpg").write_bytes(b"jpg-train-1")
    for idx, camp in enumerate(val_campaigns):
        (ds / "val" / "images" / f"val__{camp}__img{idx}.jpg").write_bytes(f"jpg-val-{camp}-{idx}".encode())
    (ds / "test" / "images" / "test__camp_vfli_cosne__f001.jpg").write_bytes(b"jpg-test-1")
    (ds / "data.yaml").write_text(
        f"path: {ds.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: open_top_freight_wagon\n",
        encoding="utf-8",
    )
    return ds


def _write_fake_train_script(path: Path) -> None:
    path.write_text(
        '''
import argparse, json, pathlib, time
p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--data", required=True)
p.add_argument("--name", required=True)
p.add_argument("--project-dir", required=True)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--patience", type=int, default=15)
p.add_argument("--imgsz", type=int, default=640)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--freeze", type=int, default=10)
p.add_argument("--lr0", type=float, default=0.0005)
p.add_argument("--cos-lr", action="store_true")
p.add_argument("--eval-test", action="store_true")
args = p.parse_args()
if args.eval_test:
    raise SystemExit("FORBIDDEN: --eval-test must never be passed from UI!")

out = pathlib.Path(args.project_dir) / args.name
(out / "weights").mkdir(parents=True, exist_ok=True)
(out / "weights" / "best.pt").write_bytes(f"pt:{args.model}:{args.name}".encode())
(out / "args_seen.json").write_text(json.dumps({"model": args.model, "epochs": args.epochs, "eval_test": args.eval_test}))
csv_path = out / "results.csv"
csv_path.write_text(
    "epoch,time,train/box_loss,train/cls_loss,train/l1_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/l1_loss,lr/pg0\\n"
    "1,1.2,1.5,2.2,0.01,0.72,0.75,0.78,0.52,1.4,2.1,0.02,0.0004\\n"
    "2,2.4,1.2,1.8,0.01,0.86,0.88,0.935,0.71,1.1,1.7,0.015,0.0003\\n",
    encoding="utf-8",
)
time.sleep(0.15)
(out / "eval_summary.json").write_text(json.dumps({
    "experiment_name": args.name,
    "hyperparameters": {"epochs_completed": 2},
    "environment": {"peak_allocated_vram_mib": 780.5, "train_wall_time_s": 2.4},
    "val_metrics": {"precision": 0.86, "recall": 0.88, "mAP50": 0.935, "mAP50_95": 0.71},
    "val_conf_calibration": {"best_f1_conf": 0.35, "curve": [{"conf": 0.25, "tp": 10, "fp": 1, "fn": 1, "precision": 0.91, "recall": 0.91, "f1": 0.91}]},
}), encoding="utf-8")
''',
        encoding="utf-8",
    )


def _make_isolated_env(tmp_path: Path) -> tuple[Settings, ModelRegistryService, VisionRuntimeManager, TrainingService, Path]:
    datasets_root = tmp_path / "datasets"
    _create_yolo_dataset(datasets_root, "open-top-yolo-v1", val_campaigns=["camp_sishen_saldanha"])
    _create_yolo_dataset(datasets_root, "open-top-yolo-v2", val_campaigns=["camp_sishen_saldanha", "camp_oberhausen"])
    # Non-open-top domain dataset should be ignored automatically
    twist = datasets_root / "twist-yolo-v1"
    (twist / "train" / "images").mkdir(parents=True, exist_ok=True)
    (twist / "val" / "images").mkdir(parents=True, exist_ok=True)
    (twist / "data.yaml").write_text("names:\n  0: empty_container\n", encoding="utf-8")

    fake_pt = tmp_path / "exp04c_best.pt"
    fake_pt.write_bytes(b"fake-exp04c-weights")
    reg_path = tmp_path / "registry.json"
    reg_path.write_text(
        json.dumps(
            {
                "schemaVersion": "1.0",
                "activeModelId": "exp04c_yolo26n_opentop_640",
                "checkpoints": [
                    {
                        "id": "exp04c_yolo26n_opentop_640",
                        "modelName": "exp04c_yolo26n_opentop_640",
                        "sourceExperiment": "exp04c_yolo26n_opentop_freeze10",
                        "architecture": "yolo26n",
                        "status": "active",
                        "inputSize": 640,
                        "classNames": {"0": "open_top_freight_wagon"},
                        "pytorchWeightsPath": str(fake_pt),
                        "valMetrics": {"map50": 0.901, "map50_95": 0.692},
                        "testMetrics": {"map50": 0.2553},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    base = get_settings()
    settings = base.model_copy(
        update={
            "storage_root_raw": str(tmp_path / "storage"),
            "onnx_runs_storage_path_raw": str(tmp_path / "lab-runs"),
            "annotation_studio_path_raw": str(tmp_path / "studio"),
            "training_runs_path_raw": str(tmp_path / "training-runs"),
            "datasets_root_raw": str(datasets_root),
            "model_registry_path_raw": str(reg_path),
            "pytorch_weights_path_raw": str(fake_pt),
        }
    )

    registry = ModelRegistryService(settings)
    runtime_mgr = VisionRuntimeManager(settings, model_registry=registry)
    fake_script = tmp_path / "fake_train.py"
    _write_fake_train_script(fake_script)

    detect_root = tmp_path / "detect-runs"
    service = TrainingService(
        settings=settings,
        model_registry=registry,
        runtime_manager=runtime_mgr,
        detect_runs_root=detect_root,
        train_script_path=fake_script,
        python_executable=sys.executable,
        require_cuda=False,
    )
    return settings, registry, runtime_mgr, service, detect_root


def test_parse_ultralytics_results_csv_with_padded_headers(tmp_path: Path) -> None:
    csv_file = tmp_path / "results.csv"
    csv_file.write_text(
        "                  epoch,         time, train/box_loss, train/cls_loss, train/l1_loss, metrics/precision(B), metrics/recall(B), metrics/mAP50(B), metrics/mAP50-95(B), val/box_loss, val/cls_loss, val/l1_loss, lr/pg0\n"
        "                      1,       12.4,         1.81,           2.45,          0.012,                0.71,             0.64,           0.73,              0.44,         1.62,         2.10,         0.019, 0.0002\n",
        encoding="utf-8",
    )
    pts = parse_results_csv(csv_file)
    assert len(pts) == 1
    assert pts[0].epoch == 1
    assert pts[0].map50 == 0.73
    assert pts[0].train_box_loss == 1.81
    assert pts[0].val_cls_loss == 2.1


def test_sequential_lineage_chaining_val_marker_and_register(tmp_path: Path) -> None:
    _, registry, _, service, detect_root = _make_isolated_env(tmp_path)

    ov0 = service.get_overview()
    assert ov0.next_version_label == "v1"
    assert len(ov0.datasets) == 2
    ds_v1 = next(d for d in ov0.datasets if d.id == "open-top-yolo-v1")
    assert ds_v1.warning is not None and "1 campanha" in ds_v1.warning
    ds_v2 = next(d for d in ov0.datasets if d.id == "open-top-yolo-v2")
    assert ds_v2.warning is None

    # Launch v1 on open-top-yolo-v1 (automatically starts from exp04c)
    run1 = service.start_run(StartTrainingRequest(dataset_id="open-top-yolo-v1", epochs=2, notes="Rodada inicial"))
    assert run1.version_label == "v1"
    for _ in range(40):
        det1 = service.get_run(run1.run_id)
        if det1.status == "COMPLETED":
            break
        time.sleep(0.1)
    assert det1.status == "COMPLETED"
    assert det1.final_map50 == 0.935
    assert det1.final_map50_95 == 0.71
    assert det1.best_map50_95 == 0.71
    det1_alias = det1.model_dump(by_alias=True)
    assert det1_alias["finalMap5095"] == 0.71
    assert det1_alias["bestMap5095"] == 0.71
    assert det1_alias["epochHistory"][0]["map5095"] == 0.52
    assert len(det1.epoch_history) == 2
    seen1 = json.loads((detect_root / run1.experiment_name / "args_seen.json").read_text())
    assert seen1["eval_test"] is False
    assert seen1["model"].endswith("exp04c_best.pt")

    # Launch v2 on open-top-yolo-v2 -> must automatically chain from v1 best.pt and flag valDatasetChanged
    ov1 = service.get_overview()
    assert ov1.next_version_label == "v2"
    assert "v1" in ov1.recommended_base_label

    run2 = service.start_run(StartTrainingRequest(dataset_id="open-top-yolo-v2", epochs=2))
    assert run2.version_label == "v2"
    for _ in range(40):
        det2 = service.get_run(run2.run_id)
        if det2.status == "COMPLETED":
            break
        time.sleep(0.1)
    assert det2.status == "COMPLETED"
    seen2 = json.loads((detect_root / run2.experiment_name / "args_seen.json").read_text())
    assert run1.experiment_name in seen2["model"]

    ov2 = service.get_overview()
    labels = [pt.version_label for pt in ov2.lineage_points]
    assert labels == ["v0", "v1", "v2"]
    assert ov2.lineage_points[2].val_dataset_changed is True
    ov2_alias = ov2.model_dump(by_alias=True)
    assert isinstance(ov2_alias["baselineMetrics"]["map5095"], float)
    assert ov2_alias["baselineMetrics"]["map5095"] == ov2.baseline_metrics.map50_95
    assert ov2_alias["lineagePoints"][1]["map5095"] == 0.71

    # Register v2 into Model Registry as candidate
    cp_dict = service.register_run_checkpoint(run2.run_id, RegisterTrainingCheckpointRequest(activate=False))
    assert cp_dict["status"] == "candidate"
    assert registry.get_checkpoint(cp_dict["id"]).val_metrics["map50"] == 0.935
    assert registry.get_checkpoint(cp_dict["id"]).val_metrics["map50_95"] == 0.71


def test_training_api_concurrency_lock_stop_and_recovery(tmp_path: Path) -> None:
    settings, _, runtime_mgr, _, detect_root = _make_isolated_env(tmp_path)
    slow_script = tmp_path / "slow_train.py"
    slow_script.write_text(
        """
import argparse, pathlib, time
p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--project-dir", required=True)
args, _ = p.parse_known_args()
out = pathlib.Path(args.project_dir) / args.name
(out / "weights").mkdir(parents=True, exist_ok=True)
(out / "weights" / "best.pt").write_bytes(b"partial-pt")
(out / "results.csv").write_text(
    "epoch,time,train/box_loss,train/cls_loss,train/l1_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/l1_loss,lr/pg0\\n"
    "1,0.5,1.6,2.1,0.01,0.6,0.6,0.65,0.4,1.5,2.0,0.02,0.0004\\n",
    encoding="utf-8",
)
time.sleep(30)
""",
        encoding="utf-8",
    )

    app = create_app(settings=settings, enable_startup_recovery=False)
    app.state.training_service = TrainingService(
        settings=settings,
        model_registry=app.state.model_registry,
        runtime_manager=app.state.runtime_manager,
        detect_runs_root=detect_root,
        train_script_path=slow_script,
        python_executable=sys.executable,
        require_cuda=False,
    )

    with TestClient(app) as client:
        r0 = client.get("/api/v1/training/overview")
        assert r0.status_code == 200

        r_start = client.post("/api/v1/training/runs", json={"datasetId": "open-top-yolo-v1", "epochs": 5})
        assert r_start.status_code == 202
        run_id = r_start.json()["runId"]

        # Second concurrent start must fail with 409 Conflict
        r_dup = client.post("/api/v1/training/runs", json={"datasetId": "open-top-yolo-v1", "epochs": 5})
        assert r_dup.status_code == 409

        time.sleep(0.4)
        r_live = client.get(f"/api/v1/training/runs/{run_id}")
        assert r_live.status_code == 200
        assert len(r_live.json()["epochHistory"]) == 1

        r_stop = client.post(f"/api/v1/training/runs/{run_id}/stop")
        assert r_stop.status_code == 200
        assert r_stop.json()["status"] == "CANCELLED"

    # Simulate unclean restart leaving a RUNNING record on disk
    stuck_id = UUID("11111111-1111-1111-1111-111111111111")
    app.state.training_service._save_run_record(
        {
            "runId": str(stuck_id),
            "versionNumber": 99,
            "versionLabel": "v99",
            "experimentName": "stuck_exp",
            "status": "RUNNING",
            "datasetId": "open-top-yolo-v1",
            "valFingerprint": "abc",
            "baseId": "auto",
            "baseLabel": "v0",
            "architecture": "yolo26n",
            "epochsRequested": 5,
            "detectOutputDir": str(detect_root / "stuck_exp"),
            "createdAt": "2026-11-24T00:00:00Z",
        }
    )
    recovered = app.state.training_service.recover_interrupted_runs()
    assert recovered == 1
    assert app.state.training_service.get_run(stuck_id).status == "FAILED"
