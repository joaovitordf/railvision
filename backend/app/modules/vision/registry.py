"""Declarative Model Registry service backed by artifacts/models/registry.json."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

from app.core.config import Settings
from app.core.exceptions import ResourceNotFoundException
from app.core.paths import REPO_ROOT, resolve_repo_path

ACTIVE_DEFAULT_MODEL_ID = "exp04c_yolo26n_opentop_640"


def _sha256_or_none(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


@dataclass(frozen=True, slots=True)
class RegisteredCheckpoint:
    """Single registered open-top wagon detection checkpoint entry."""

    id: str
    model_name: str
    source_experiment: str
    architecture: str
    status: str
    input_size: int
    class_names: dict[int, str]
    pytorch_weights_path: Path | None
    pytorch_weights_sha256: str | None
    onnx_model_path: Path | None
    onnx_meta_path: Path | None
    onnx_model_sha256: str | None
    limitations_notice: str
    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize checkpoint into JSON-compatible dictionary with repo-relative paths."""
        return {
            "id": self.id,
            "modelName": self.model_name,
            "sourceExperiment": self.source_experiment,
            "architecture": self.architecture,
            "status": self.status,
            "inputSize": self.input_size,
            "classNames": {str(k): v for k, v in self.class_names.items()},
            "pytorchWeightsPath": _to_repo_rel(self.pytorch_weights_path),
            "pytorchWeightsSha256": self.pytorch_weights_sha256,
            "onnxModelPath": _to_repo_rel(self.onnx_model_path),
            "onnxMetaPath": _to_repo_rel(self.onnx_meta_path),
            "onnxModelSha256": self.onnx_model_sha256,
            "limitationsNotice": self.limitations_notice,
            "valMetrics": dict(self.val_metrics),
            "testMetrics": dict(self.test_metrics),
        }


def _checkpoint_from_dict(raw: dict[str, Any]) -> RegisteredCheckpoint:
    pt_raw = raw.get("pytorchWeightsPath")
    onnx_raw = raw.get("onnxModelPath")
    meta_raw = raw.get("onnxMetaPath")
    pt_path = resolve_repo_path(str(pt_raw)) if pt_raw else None
    onnx_path = resolve_repo_path(str(onnx_raw)) if onnx_raw else None
    meta_path = resolve_repo_path(str(meta_raw)) if meta_raw else None

    raw_classes = raw.get("classNames") or {"0": "open_top_freight_wagon"}
    class_names = {int(k): str(v) for k, v in raw_classes.items()}

    return RegisteredCheckpoint(
        id=str(raw["id"]),
        model_name=str(raw.get("modelName") or raw["id"]),
        source_experiment=str(raw.get("sourceExperiment") or raw["id"]),
        architecture=str(raw.get("architecture") or "yolo26n"),
        status=str(raw.get("status") or "candidate"),
        input_size=int(raw.get("inputSize") or 640),
        class_names=class_names,
        pytorch_weights_path=pt_path,
        pytorch_weights_sha256=(
            str(raw["pytorchWeightsSha256"]) if raw.get("pytorchWeightsSha256") else _sha256_or_none(pt_path)
        ),
        onnx_model_path=onnx_path,
        onnx_meta_path=meta_path,
        onnx_model_sha256=(
            str(raw["onnxModelSha256"]) if raw.get("onnxModelSha256") else _sha256_or_none(onnx_path)
        ),
        limitations_notice=str(
            raw.get("limitationsNotice")
            or "Checkpoint experimental; validar no conjunto de validação congelado antes de qualquer promoção."
        ),
        val_metrics={str(k): float(v) for k, v in (raw.get("valMetrics") or {}).items()},
        test_metrics={str(k): float(v) for k, v in (raw.get("testMetrics") or {}).items()},
    )


class ModelRegistryService:
    """Manages versioned detection checkpoints under artifacts/models/registry.json."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._registry_path = settings.model_registry_path
        self._lock = threading.Lock()
        self._entries: dict[str, RegisteredCheckpoint] = {}
        self.reload()

    @property
    def registry_path(self) -> Path:
        """Resolved filesystem path to registry.json."""
        return self._registry_path

    def reload(self) -> None:
        """Load registry entries from disk or bootstrap default open-top PoC checkpoints."""
        loaded: dict[str, RegisteredCheckpoint] = {}
        if self._registry_path.is_file():
            payload = json.loads(self._registry_path.read_text(encoding="utf-8"))
            for item in payload.get("checkpoints") or []:
                cp = _checkpoint_from_dict(item)
                loaded[cp.id] = cp
        else:
            for cp in self._build_initial_checkpoints():
                loaded[cp.id] = cp
            self._persist(loaded)

        self._discover_open_top_experiments(loaded)
        self._entries = loaded

    def _build_initial_checkpoints(self) -> list[RegisteredCheckpoint]:
        pt_default = self._settings.pytorch_weights_path
        onnx_default = self._settings.onnx_model_path
        meta_default = self._settings.onnx_model_metadata_path
        default_exp = "exp04c_yolo26n_opentop_freeze10"
        if meta_default.is_file():
            meta_json = json.loads(meta_default.read_text(encoding="utf-8"))
            model_name = str(meta_json.get("modelName") or ACTIVE_DEFAULT_MODEL_ID)
            default_exp = str(meta_json.get("sourceExperiment") or default_exp)
            input_size = int(meta_json.get("imgsz") or 640)
            limitations = str(
                meta_json.get("limitationsNotice")
                or "PoC model (exp04c): good validation score on vid_15, limited generalization on test vid_09/vid_16. Not approved for production."
            )
            onnx_sha = str(meta_json.get("onnxSha256") or "") or _sha256_or_none(onnx_default)
        else:
            model_name = ACTIVE_DEFAULT_MODEL_ID
            input_size = 640
            limitations = (
                "PoC model (exp04c): bom desempenho em validação, generalização limitada no teste. Não aprovado para produção."
            )
            onnx_sha = _sha256_or_none(onnx_default)

        entries = [
            RegisteredCheckpoint(
                id=ACTIVE_DEFAULT_MODEL_ID,
                model_name=model_name,
                source_experiment=default_exp,
                architecture="yolo26n",
                status="active",
                input_size=input_size,
                class_names={0: "open_top_freight_wagon"},
                pytorch_weights_path=pt_default if pt_default.is_file() else None,
                pytorch_weights_sha256=_sha256_or_none(pt_default),
                onnx_model_path=onnx_default if onnx_default.is_file() else None,
                onnx_meta_path=meta_default if meta_default.is_file() else None,
                onnx_model_sha256=onnx_sha,
                limitations_notice=limitations,
                val_metrics={"map50": 0.901, "map50_95": 0.692},
                test_metrics={"map50": 0.2553, "map50_95": 0.1456},
            )
        ]
        self._discover_open_top_experiments_into_list(entries)
        return entries

    def _discover_open_top_experiments_into_list(self, items: list[RegisteredCheckpoint]) -> None:
        known_ids = {item.id for item in items}
        detect_root = REPO_ROOT / "artifacts" / "runs" / "detect"
        if not detect_root.is_dir():
            return
        for exp_dir in sorted(detect_root.iterdir()):
            if not exp_dir.is_dir() or "opentop" not in exp_dir.name.lower():
                continue
            weights_pt = exp_dir / "weights" / "best.pt"
            if not weights_pt.is_file():
                continue
            if exp_dir.name == "exp04c_yolo26n_opentop_freeze10" or exp_dir.name in known_ids:
                continue
            arch = "yolo26s" if "yolo26s" in exp_dir.name.lower() else "yolo26n"
            items.append(
                RegisteredCheckpoint(
                    id=exp_dir.name,
                    model_name=exp_dir.name,
                    source_experiment=exp_dir.name,
                    architecture=arch,
                    status="archived",
                    input_size=640,
                    class_names={0: "open_top_freight_wagon"},
                    pytorch_weights_path=weights_pt,
                    pytorch_weights_sha256=_sha256_or_none(weights_pt),
                    onnx_model_path=None,
                    onnx_meta_path=None,
                    onnx_model_sha256=None,
                    limitations_notice=f"Checkpoint arquivado ({exp_dir.name}) disponível apenas para comparação no Vision Lab.",
                )
            )

    def _discover_open_top_experiments(self, entries: dict[str, RegisteredCheckpoint]) -> None:
        known_exps = {item.source_experiment for item in entries.values()}
        detect_root = REPO_ROOT / "artifacts" / "runs" / "detect"
        if not detect_root.is_dir():
            return
        added = False
        for exp_dir in sorted(detect_root.iterdir()):
            if not exp_dir.is_dir() or "opentop" not in exp_dir.name.lower():
                continue
            if exp_dir.name in known_exps or exp_dir.name in entries:
                continue
            weights_pt = exp_dir / "weights" / "best.pt"
            if not weights_pt.is_file():
                continue
            arch = "yolo26s" if "yolo26s" in exp_dir.name.lower() else "yolo26n"
            entries[exp_dir.name] = RegisteredCheckpoint(
                id=exp_dir.name,
                model_name=exp_dir.name,
                source_experiment=exp_dir.name,
                architecture=arch,
                status="archived",
                input_size=640,
                class_names={0: "open_top_freight_wagon"},
                pytorch_weights_path=weights_pt,
                pytorch_weights_sha256=_sha256_or_none(weights_pt),
                onnx_model_path=None,
                onnx_meta_path=None,
                onnx_model_sha256=None,
                limitations_notice=f"Checkpoint arquivado ({exp_dir.name}) disponível apenas para comparação no Vision Lab.",
            )
            added = True
        if added and self._registry_path.is_file():
            self._persist(entries)

    def _persist(self, entries: dict[str, RegisteredCheckpoint]) -> None:
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        active_id = next(
            (cp.id for cp in entries.values() if cp.status == "active"),
            ACTIVE_DEFAULT_MODEL_ID,
        )
        payload = {
            "schemaVersion": "1.0",
            "activeModelId": active_id,
            "checkpoints": [cp.to_dict() for cp in entries.values()],
        }
        tmp = self._registry_path.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._registry_path)

    def register_checkpoint(self, checkpoint: RegisteredCheckpoint) -> RegisteredCheckpoint:
        """Persist or update a checkpoint entry in registry.json in a thread-safe manner."""
        with self._lock:
            self._entries[checkpoint.id] = checkpoint
            self._persist(self._entries)
            return checkpoint

    def set_active_checkpoint(self, model_id: str) -> RegisteredCheckpoint:
        """Promote a registered checkpoint to 'active', demoting the previous active entry to 'candidate'."""
        with self._lock:
            if model_id not in self._entries:
                raise ResourceNotFoundException(f"Checkpoint não registrado no Model Registry: {model_id}")
            updated: dict[str, RegisteredCheckpoint] = {}
            for cid, entry in self._entries.items():
                if cid == model_id:
                    updated[cid] = replace(entry, status="active")
                elif entry.status == "active":
                    updated[cid] = replace(entry, status="candidate")
                else:
                    updated[cid] = entry
            self._entries = updated
            self._persist(self._entries)
            return self._entries[model_id]

    def list_checkpoints(self) -> list[RegisteredCheckpoint]:
        """Return registered open-top checkpoints ordered with active first."""
        priority = {"active": 0, "candidate": 1, "archived": 2}
        with self._lock:
            items = list(self._entries.values())
        return sorted(
            items,
            key=lambda item: (priority.get(item.status, 9), item.id),
        )

    def get_checkpoint(self, model_id: str) -> RegisteredCheckpoint:
        """Fetch a checkpoint by ID or raise ResourceNotFoundException."""
        with self._lock:
            cp = self._entries.get(model_id)
        if cp is None:
            raise ResourceNotFoundException(f"Checkpoint não registrado no Model Registry: {model_id}")
        return cp

    def get_active_checkpoint(self) -> RegisteredCheckpoint:
        """Return the active default checkpoint."""
        with self._lock:
            items = list(self._entries.values())
        for cp in items:
            if cp.status == "active":
                return cp
        return items[0]
