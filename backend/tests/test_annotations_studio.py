"""Backend tests for the native Annotation Studio MVP (/api/v1/annotations/*)."""

from __future__ import annotations

import io
import json
from pathlib import Path
import zipfile

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.core.config import get_settings
from app.core.paths import REPO_ROOT
from app.main import create_app


def _make_app(tmp_path: Path):
    settings = get_settings().model_copy(
        update={
            "storage_root_raw": str(tmp_path / "app-storage"),
            "onnx_runs_storage_path_raw": str(tmp_path / "lab-runs"),
            "curation_inbox_path_raw": str(tmp_path / "curation-inbox"),
            "annotation_studio_path_raw": str(tmp_path / "annotation-studio"),
        }
    )
    return create_app(settings, enable_startup_recovery=False), settings


def _make_jpg_bytes(width: int = 320, height: int = 240, color: tuple[int, int, int] = (40, 120, 200)) -> bytes:
    img = np.full((height, width, 3), color, dtype=np.uint8)
    ok, enc = cv2.imencode(".jpg", img)
    assert ok
    return enc.tobytes()


@pytest.mark.unit
def test_annotations_project_crud_upload_review_and_yolo_export_import(tmp_path: Path) -> None:
    """Verify full project -> multi-image upload -> human review -> YOLO 1.1 zip gate & roundtrip."""
    app, settings = _make_app(tmp_path)
    with TestClient(app) as client:
        # 1. Create project
        create_resp = client.post(
            "/api/v1/annotations/projects",
            json={
                "name": "Campanha Teste",
                "campaignId": "camp_test_01",
                "sourceOrigin": "proprio",
                "description": "Revisão inicial",
            },
        )
        assert create_resp.status_code == 201
        project = create_resp.json()
        pid = project["id"]
        assert project["classes"] == ["open_top_freight_wagon"]
        assert project["itemCount"] == 0

        # 2. Reserved benchmark filename warning
        up_resp = client.post(
            f"/api/v1/annotations/projects/{pid}/images",
            files=[
                ("files", ("frame_ok.jpg", _make_jpg_bytes(320, 240), "image/jpeg")),
                ("files", ("vid_01_frame001.jpg", _make_jpg_bytes(320, 240, (10, 20, 30)), "image/jpeg")),
            ],
        )
        assert up_resp.status_code == 201
        items = up_resp.json()
        assert len(items) == 2
        assert items[0]["reservedBenchmarkWarning"] is False
        assert items[1]["reservedBenchmarkWarning"] is True
        item_ok = items[0]

        # 3. Serve clean image and thumbnail
        assert client.get(item_ok["imageUrl"]).status_code == 200
        assert client.get(item_ok["thumbnailUrl"]).status_code == 200

        # 4. Export YOLO blocked with 409 before human review
        blocked = client.post(
            f"/api/v1/annotations/projects/{pid}/export-yolo",
            json={"onlyHumanReviewed": True, "includeUnreviewedDrafts": False},
        )
        assert blocked.status_code == 409

        # 5. Save human-reviewed box on item 0 + mark item 1 rejected
        put_resp = client.put(
            f"/api/v1/annotations/projects/{pid}/items/{item_ok['id']}/annotations",
            json={
                "boxes": [
                    {
                        "id": "b1",
                        "classId": 0,
                        "className": "open_top_freight_wagon",
                        "x": 0.1,
                        "y": 0.2,
                        "width": 0.5,
                        "height": 0.4,
                        "origin": "manual",
                    }
                ],
                "noObjectsConfirmed": False,
                "status": "approved",
                "reviewNotes": "Caixa validada",
                "falsePositivesRemoved": 1,
                "falseNegativesAdded": 1,
            },
        )
        assert put_resp.status_code == 200
        saved = put_resp.json()
        assert saved["humanReviewed"] is True
        assert saved["status"] == "approved"
        assert saved["approvedAt"] is not None

        rej_resp = client.put(
            f"/api/v1/annotations/projects/{pid}/items/{items[1]['id']}/annotations",
            json={"boxes": [], "noObjectsConfirmed": True, "status": "rejected"},
        )
        assert rej_resp.status_code == 200

        # 6. Export YOLO zip -> contains only approved item, excludes rejected
        exp_resp = client.post(
            f"/api/v1/annotations/projects/{pid}/export-yolo",
            json={"onlyHumanReviewed": True, "includeUnreviewedDrafts": False},
        )
        assert exp_resp.status_code == 200
        assert exp_resp.headers["content-type"] == "application/zip"
        with zipfile.ZipFile(io.BytesIO(exp_resp.content), mode="r") as zf:
            names = set(zf.namelist())
            assert {"obj.names", "obj.data", "train.txt", "provenance.json"}.issubset(names)
            prov = json.loads(zf.read("provenance.json"))
            assert prov["humanReviewed"] is True
            assert len(prov["items"]) == 1
            assert prov["items"][0]["sourceFilename"] == "frame_ok.jpg"

        # 7. Re-import exported zip into a fresh project
        p2_id = client.post(
            "/api/v1/annotations/projects",
            json={"name": "Reimport", "campaignId": "camp_reimport", "sourceOrigin": "cvat"},
        ).json()["id"]
        imp_resp = client.post(
            f"/api/v1/annotations/projects/{p2_id}/import-yolo",
            files={"file": ("export.zip", exp_resp.content, "application/zip")},
        )
        assert imp_resp.status_code == 201
        imp_items = imp_resp.json()
        assert len(imp_items) == 1
        assert imp_items[0]["boxCount"] == 1
        assert imp_items[0]["humanReviewed"] is False

        # 8. Ensure official datasets were never touched
        assert (REPO_ROOT / "data/processed/open-top-yolo-v1").is_dir()
        assert (settings.annotation_studio_path / pid / "project.json").is_file()

        # 9. Delete project cleanly
        del_resp = client.delete(f"/api/v1/annotations/projects/{pid}")
        assert del_resp.status_code == 204
        assert client.get(f"/api/v1/annotations/projects/{pid}").status_code == 404


@pytest.mark.gpu
def test_annotations_ogv_dedup_extraction_preannotate_and_inbox_import(tmp_path: Path) -> None:
    """Verify .ogv video upload, deduplicated preview/extract, real exp04c preannotation, and curation-inbox import."""
    vid15 = REPO_ROOT / "data/raw/public-videos/vid_15_sishen_saldanha_railway_line.ogv"
    val_img = REPO_ROOT / "data/processed/open-top-yolo-v1/val/images/val__camp_sishen_saldanha__vid_15__f000075.jpg"
    if not vid15.is_file() or not val_img.is_file():
        pytest.skip("Real vid_15 fixtures not available.")

    app, settings = _make_app(tmp_path)
    with TestClient(app) as client:
        pid = client.post(
            "/api/v1/annotations/projects",
            json={"name": "Sishen Studio", "campaignId": "camp_sishen_saldanha", "sourceOrigin": "public"},
        ).json()["id"]

        # 1. Upload .ogv video and preview deduplicated frames
        with vid15.open("rb") as handle:
            vid_resp = client.post(
                f"/api/v1/annotations/projects/{pid}/videos",
                files={"file": (vid15.name, handle.read(), "video/ogg")},
            )
        assert vid_resp.status_code == 201
        vid = vid_resp.json()
        assert vid["codec"] == "theora"

        prev_resp = client.post(
            f"/api/v1/annotations/projects/{pid}/videos/{vid['id']}/preview-frames",
            json={"sampleFps": 1.0, "maxFrames": 6, "dedupThreshold": 3.0},
        )
        assert prev_resp.status_code == 200
        prev = prev_resp.json()
        assert len(prev["candidates"]) >= 2
        first_cand = prev["candidates"][0]
        assert client.get(first_cand["previewUrl"]).status_code == 200

        # 2. Extract first 2 selected frames at full resolution
        ext_resp = client.post(
            f"/api/v1/annotations/projects/{pid}/videos/{vid['id']}/extract-frames",
            json={"frameIndices": [c["frameIndex"] for c in prev["candidates"][:2]]},
        )
        assert ext_resp.status_code == 201
        ext_items = ext_resp.json()
        assert len(ext_items) == 2
        assert ext_items[0]["sourceVideoFilename"] == vid15.name
        assert ext_items[0]["widthPx"] == 1920

        # 3. Upload a known positive val frame and run batch pre-annotation with exp04c on GPU
        up = client.post(
            f"/api/v1/annotations/projects/{pid}/images",
            files=[("files", (val_img.name, val_img.read_bytes(), "image/jpeg"))],
        ).json()[0]

        batch = client.post(
            f"/api/v1/annotations/projects/{pid}/preannotate-batch",
            json={"modelId": "exp04c_yolo26n_opentop_640", "runtime": "pytorch_cuda", "conf": 0.25, "iou": 0.45},
        )
        assert batch.status_code == 200
        batch_data = batch.json()
        assert batch_data["processedItems"] == 3
        assert batch_data["totalSuggestions"] >= 1

        detail = client.get(f"/api/v1/annotations/projects/{pid}/items/{up['id']}").json()
        assert detail["status"] == "pre_annotated"
        assert detail["humanReviewed"] is False
        assert detail["preannotationModelId"] == "exp04c_yolo26n_opentop_640"
        assert len(detail["boxes"]) >= 1
        assert all(b["origin"] == "ai_suggestion" for b in detail["boxes"])

        # 4. Seed curation-inbox item and import into project cleanly
        inbox_day = settings.curation_inbox_path / "2026-11-24"
        inbox_day.mkdir(parents=True, exist_ok=True)
        (inbox_day / "smoke_f000001.jpg").write_bytes(_make_jpg_bytes(320, 240, (90, 90, 90)))
        (inbox_day / "smoke_f000001.txt").write_text("0 0.5 0.5 0.3 0.2\n", encoding="utf-8")
        (inbox_day / "smoke_f000001.meta.json").write_text(
            json.dumps({"sourceFilename": "curated_smoke.jpg", "frameIndex": 1, "timestampMs": 200}),
            encoding="utf-8",
        )
        inbox_imp = client.post(f"/api/v1/annotations/projects/{pid}/import-curation-inbox").json()
        assert inbox_imp["importedCount"] == 1
        assert inbox_imp["items"][0]["boxCount"] == 1
        # Second import skips duplicate SHA-256
        inbox_imp2 = client.post(f"/api/v1/annotations/projects/{pid}/import-curation-inbox").json()
        assert inbox_imp2["importedCount"] == 0
        assert inbox_imp2["skippedDuplicates"] == 1


@pytest.mark.unit
def test_import_yolo_zip_handles_malformed_labels_and_rejects_traversal(tmp_path: Path) -> None:
    """Ensure YOLO zip import ignores traversal entries and malformed label lines without HTTP 500."""
    app, _ = _make_app(tmp_path)
    with TestClient(app) as client:
        pid = client.post(
            "/api/v1/annotations/projects",
            json={"name": "Zip Security Check", "campaignId": "camp_zip_sec", "sourceOrigin": "proprio"},
        ).json()["id"]

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("../evil.jpg", _make_jpg_bytes(160, 120))
            zf.writestr("obj_train_data/good_frame.jpg", _make_jpg_bytes(320, 240))
            zf.writestr(
                "obj_train_data/good_frame.txt",
                "bad line x y z\n0 not_a_float 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n",
            )

        resp = client.post(
            f"/api/v1/annotations/projects/{pid}/import-yolo",
            files={"file": ("labels.zip", buf.getvalue(), "application/zip")},
        )
        assert resp.status_code == 201
        imported = resp.json()
        assert len(imported) == 1
        assert imported[0]["filename"] == "good_frame.jpg"
        assert imported[0]["boxCount"] == 1
