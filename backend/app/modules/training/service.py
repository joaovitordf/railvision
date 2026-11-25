"""Managed YOLO training orchestrator with versioned lineage, live CSV telemetry, and GPU locking."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any
from uuid import UUID, uuid4

import torch

from app.core.config import Settings
from app.core.exceptions import (
    ConflictException,
    InvalidMediaException,
    ResourceNotFoundException,
    RuntimeUnavailableException,
)
from app.core.paths import REPO_ROOT, resolve_repo_path
from app.modules.training.schemas import (
    CalibrationCurvePointDto,
    RegisterTrainingCheckpointRequest,
    StartTrainingRequest,
    TrainingBaseOptionDto,
    TrainingBaselineMetricsDto,
    TrainingDatasetOptionDto,
    TrainingEpochPointDto,
    TrainingLineagePointDto,
    TrainingOverviewDto,
    TrainingRunDetailDto,
    TrainingRunSummaryDto,
)
from app.modules.vision.registry import ACTIVE_DEFAULT_MODEL_ID, ModelRegistryService, RegisteredCheckpoint
from app.modules.vision.runtimes import VisionRuntimeManager

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_PRETRAINED_BASES = {"yolo26n.pt", "yolo26s.pt"}
RESERVED_BASE_DIR_TOKENS = ("exp01_", "exp02_", "exp03_", "exp04a_", "exp04b_", "exp04c_")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        val = float(str(raw).strip())
        return val if val == val else default  # guard NaN
    except (TypeError, ValueError):
        return default


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_yolo_data_yaml_names(raw_yaml: str) -> list[str]:
    """Extract class names from a Ultralytics data.yaml file without requiring PyYAML."""
    names: list[str] = []
    in_names_block = False
    names_indent = -1

    for raw_line in raw_yaml.splitlines():
        no_comment = re.sub(r"\s+#.*$", "", raw_line)
        stripped = no_comment.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(no_comment) - len(no_comment.lstrip(" "))

        if not in_names_block:
            match = re.match(r"^names\s*:\s*(.*)$", stripped)
            if not match:
                continue
            rhs = match.group(1).strip()
            if rhs.startswith("[") and rhs.endswith("]"):
                inner = rhs[1:-1].strip()
                if inner:
                    for item in inner.split(","):
                        val = item.strip().strip("\"'")
                        if val:
                            names.append(val)
                return names
            if rhs:
                names.append(rhs.strip("\"'"))
                return names
            in_names_block = True
            names_indent = indent
            continue

        if indent <= names_indent and not stripped.startswith("-"):
            break

        if stripped.startswith("-"):
            val = stripped[1:].strip().strip("\"'")
            if val:
                names.append(val)
        elif ":" in stripped:
            _, rhs = stripped.split(":", 1)
            val = rhs.strip().strip("\"'")
            if val:
                names.append(val)

    return names


def parse_results_csv(csv_path: Path) -> list[TrainingEpochPointDto]:
    """Parse Ultralytics results.csv safely even while actively being written."""
    if not csv_path.is_file():
        return []
    try:
        raw_text = csv_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    lines = [ln for ln in raw_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []

    reader = csv.DictReader(lines)
    if not reader.fieldnames:
        return []

    points: list[TrainingEpochPointDto] = []
    for raw_row in reader:
        row = {str(k).strip(): str(v).strip() for k, v in raw_row.items() if k is not None}
        if not row.get("epoch"):
            continue
        try:
            epoch_num = int(float(row["epoch"]))
        except ValueError:
            continue
        train_l1 = row.get("train/l1_loss") if row.get("train/l1_loss") not in (None, "") else row.get("train/dfl_loss")
        val_l1 = row.get("val/l1_loss") if row.get("val/l1_loss") not in (None, "") else row.get("val/dfl_loss")
        points.append(
            TrainingEpochPointDto(
                epoch=epoch_num,
                time_sec=round(_safe_float(row.get("time")), 2),
                train_box_loss=round(_safe_float(row.get("train/box_loss")), 5),
                train_cls_loss=round(_safe_float(row.get("train/cls_loss")), 5),
                train_l1_loss=round(_safe_float(train_l1), 5),
                val_box_loss=round(_safe_float(row.get("val/box_loss")), 5),
                val_cls_loss=round(_safe_float(row.get("val/cls_loss")), 5),
                val_l1_loss=round(_safe_float(val_l1), 5),
                precision=round(_safe_float(row.get("metrics/precision(B)")), 4),
                recall=round(_safe_float(row.get("metrics/recall(B)")), 4),
                map50=round(_safe_float(row.get("metrics/mAP50(B)")), 4),
                map50_95=round(_safe_float(row.get("metrics/mAP50-95(B)")), 4),
                lr=_safe_float(row.get("lr/pg0"), default=-1.0) if row.get("lr/pg0") else None,
            )
        )
    return sorted(points, key=lambda p: p.epoch)


class TrainingService:
    """Orchestrates incremental YOLO training runs, GPU exclusivity, and lineage tracking."""

    def __init__(
        self,
        *,
        settings: Settings,
        model_registry: ModelRegistryService,
        runtime_manager: VisionRuntimeManager,
        detect_runs_root: Path | None = None,
        train_script_path: Path | None = None,
        python_executable: str | None = None,
        require_cuda: bool = True,
    ) -> None:
        self._settings = settings
        self._registry = model_registry
        self._runtime_manager = runtime_manager
        self._runs_root = settings.training_runs_path
        self._runs_root.mkdir(parents=True, exist_ok=True)
        self._detect_runs_root = detect_runs_root or (REPO_ROOT / "artifacts" / "runs" / "detect")
        self._train_script = train_script_path or (REPO_ROOT / "ml" / "scripts" / "train_detector.py")
        self._python_exe = python_executable or sys.executable
        self._require_cuda = require_cuda
        self._lock = threading.Lock()
        self._active_proc: subprocess.Popen[str] | None = None
        self._active_run_id: UUID | None = None
        self._stop_requested: set[UUID] = set()
        self._shutting_down = False
        self._worker_threads: list[threading.Thread] = []

    def _run_dir(self, run_id: UUID | str) -> Path:
        return self._runs_root / str(run_id)

    def _run_meta_path(self, run_id: UUID | str) -> Path:
        return self._run_dir(run_id) / "run.json"

    def _save_run_record(self, record: dict[str, Any]) -> None:
        run_dir = self._run_dir(record["runId"])
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / "run.json"
        tmp = run_dir / "run.tmp.json"
        tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(meta_path)

    def _load_run_record(self, run_id: UUID | str) -> dict[str, Any]:
        meta_path = self._run_meta_path(run_id)
        if not meta_path.is_file():
            raise ResourceNotFoundException(f"Execução de treinamento não encontrada: {run_id}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _list_raw_records(self) -> list[dict[str, Any]]:
        if not self._runs_root.is_dir():
            return []
        records: list[dict[str, Any]] = []
        for entry in self._runs_root.iterdir():
            if not entry.is_dir():
                continue
            meta_file = entry / "run.json"
            if not meta_file.is_file():
                continue
            try:
                records.append(json.loads(meta_file.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning("Ignorando run.json inválido em %s: %s", entry, exc)
        return sorted(records, key=lambda r: str(r.get("createdAt") or ""), reverse=True)

    @staticmethod
    def _dataset_val_fingerprint(val_images_dir: Path) -> tuple[list[Path], str]:
        files = sorted(
            [p for p in val_images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda p: p.name,
        )
        hasher = hashlib.sha256()
        for f in files:
            hasher.update(f.name.encode("utf-8"))
            hasher.update(str(f.stat().st_size).encode("utf-8"))
        return files, hasher.hexdigest()[:16]

    def discover_datasets(self) -> list[TrainingDatasetOptionDto]:
        """Discover valid open_top_freight_wagon YOLO datasets under settings.datasets_root."""
        root = self._settings.datasets_root
        if not root.is_dir():
            return []

        options: list[TrainingDatasetOptionDto] = []
        for candidate in sorted(root.iterdir()):
            if not candidate.is_dir():
                continue
            yaml_path = candidate / "data.yaml"
            train_img_dir = candidate / "train" / "images"
            val_img_dir = candidate / "val" / "images"
            if not (yaml_path.is_file() and train_img_dir.is_dir() and val_img_dir.is_dir()):
                continue
            try:
                class_values = _parse_yolo_data_yaml_names(yaml_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "open_top_freight_wagon" not in class_values:
                continue

            train_files = [
                p for p in train_img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            val_files, val_fp = self._dataset_val_fingerprint(val_img_dir)
            campaigns: set[str] = set()
            for vf in val_files:
                parts = vf.stem.split("__")
                if len(parts) >= 2 and parts[1]:
                    campaigns.add(parts[1])
                else:
                    campaigns.add("campanha_padrao")

            val_campaigns = sorted(campaigns)
            test_img_dir = candidate / "test" / "images"
            has_frozen_test = test_img_dir.is_dir() and any(
                p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in test_img_dir.iterdir()
            )
            warning: str | None = None
            if len(val_campaigns) < 2:
                warning = (
                    f"Validação atual possui apenas {len(val_campaigns)} campanha ({', '.join(val_campaigns)}). "
                    "Útil para validar o fluxo e evolução inicial, porém requer >= 2 campanhas para medir generalização real no Portão G2."
                )

            options.append(
                TrainingDatasetOptionDto(
                    id=candidate.name,
                    name=candidate.name,
                    data_yaml_path=_to_repo_rel(yaml_path),
                    train_count=len(train_files),
                    val_count=len(val_files),
                    val_campaigns=val_campaigns,
                    val_fingerprint=val_fp,
                    has_frozen_test=has_frozen_test,
                    warning=warning,
                )
            )
        return options

    def _baseline_metrics(self) -> TrainingBaselineMetricsDto:
        cp = self._registry.get_active_checkpoint()
        summary_path = self._detect_runs_root / cp.source_experiment / "eval_summary.json"
        if not summary_path.is_file():
            summary_path = REPO_ROOT / "artifacts" / "runs" / "detect" / "exp04c_yolo26n_opentop_freeze10" / "eval_summary.json"
        map50 = cp.val_metrics.get("map50")
        map50_95 = cp.val_metrics.get("map50_95")
        precision: float | None = None
        recall: float | None = None
        if summary_path.is_file():
            try:
                raw = json.loads(summary_path.read_text(encoding="utf-8"))
                vm = raw.get("val_metrics") or {}
                map50 = float(vm["mAP50"]) if vm.get("mAP50") is not None else map50
                map50_95 = float(vm["mAP50_95"]) if vm.get("mAP50_95") is not None else map50_95
                precision = float(vm["precision"]) if vm.get("precision") is not None else None
                recall = float(vm["recall"]) if vm.get("recall") is not None else None
            except Exception:
                pass
        return TrainingBaselineMetricsDto(
            model_id=cp.id,
            version_label="v0",
            source_experiment=cp.source_experiment,
            map50=map50,
            map50_95=map50_95,
            precision=precision,
            recall=recall,
        )

    def _next_version_and_recommended_base(
        self, records: list[dict[str, Any]]
    ) -> tuple[int, str, str, str]:
        max_ver = 0
        latest_completed: dict[str, Any] | None = None
        for rec in sorted(records, key=lambda r: int(r.get("versionNumber") or 0)):
            ver = int(rec.get("versionNumber") or 0)
            if ver > max_ver:
                max_ver = ver
            if rec.get("status") == "COMPLETED" and rec.get("bestWeightsPath"):
                weights_p = resolve_repo_path(str(rec["bestWeightsPath"]))
                if weights_p.is_file():
                    latest_completed = rec

        next_num = max_ver + 1
        next_label = f"v{next_num}"
        if latest_completed is not None:
            rec_id = f"run:{latest_completed['runId']}"
            rec_label = (
                f"Última versão concluída ({latest_completed['versionLabel']} · {latest_completed['experimentName']})"
            )
            return next_num, next_label, rec_id, rec_label

        active_cp = self._registry.get_active_checkpoint()
        return (
            next_num,
            next_label,
            f"checkpoint:{active_cp.id}",
            f"Baseline inicial (v0 · {active_cp.id})",
        )

    def _build_advanced_base_options(self, recommended_id: str, recommended_label: str) -> list[TrainingBaseOptionDto]:
        options = [
            TrainingBaseOptionDto(
                id="auto",
                label=f"Automático — {recommended_label}",
                kind="auto",
                description="Continua automaticamente a partir do melhor peso mais recente desta linha de versões.",
            )
        ]
        for cp in self._registry.list_checkpoints():
            if cp.pytorch_weights_path and cp.pytorch_weights_path.is_file():
                options.append(
                    TrainingBaseOptionDto(
                        id=f"checkpoint:{cp.id}",
                        label=f"{cp.id} [{cp.status}]",
                        kind="checkpoint",
                        description=f"Checkpoint registrado ({cp.architecture}, {cp.input_size}px).",
                    )
                )
        options.append(
            TrainingBaseOptionDto(
                id="pretrained:yolo26n.pt",
                label="YOLO26n (pesos base COCO)",
                kind="pretrained",
                description="Reinicia o ajuste fino diretamente dos pesos pré-treinados COCO (modelo nano, rápido).",
            )
        )
        options.append(
            TrainingBaseOptionDto(
                id="pretrained:yolo26s.pt",
                label="YOLO26s (pesos base COCO)",
                kind="pretrained",
                description="Reinicia a linha com maior capacidade (modelo small, mais VRAM e tempo por época).",
            )
        )
        return options

    def _resolve_base_weights(
        self,
        base_override: str | None,
        records: list[dict[str, Any]],
    ) -> tuple[str, str, str, str]:
        """Return (resolved_base_id, resolved_base_label, weights_arg_for_cli, architecture)."""
        _, _, rec_id, rec_label = self._next_version_and_recommended_base(records)
        chosen = (base_override or "auto").strip()
        if chosen == "auto":
            chosen = rec_id

        if chosen.startswith("run:"):
            target_run_id = chosen.split(":", 1)[1]
            rec = self._load_run_record(target_run_id)
            if rec.get("status") != "COMPLETED" or not rec.get("bestWeightsPath"):
                raise InvalidMediaException(
                    "INVALID_TRAINING_BASE",
                    f"Execução base '{target_run_id}' ainda não concluiu ou não possui best.pt.",
                )
            weights_path = resolve_repo_path(str(rec["bestWeightsPath"]))
            if not weights_path.is_file():
                raise ResourceNotFoundException(f"Pesos base não encontrados em disco: {weights_path}")
            return (
                chosen,
                f"{rec['versionLabel']} ({rec['experimentName']})",
                str(weights_path),
                str(rec.get("architecture") or "yolo26n"),
            )

        if chosen.startswith("checkpoint:"):
            cp_id = chosen.split(":", 1)[1]
            cp = self._registry.get_checkpoint(cp_id)
            if cp.pytorch_weights_path is None or not cp.pytorch_weights_path.is_file():
                raise ResourceNotFoundException(f"Pesos PyTorch ausentes para checkpoint '{cp_id}'.")
            return (
                chosen,
                f"{cp.id}",
                str(cp.pytorch_weights_path.resolve()),
                cp.architecture,
            )

        if chosen.startswith("pretrained:"):
            pt_name = chosen.split(":", 1)[1]
            if pt_name not in SUPPORTED_PRETRAINED_BASES:
                raise InvalidMediaException(
                    "INVALID_TRAINING_BASE",
                    f"Modelo pré-treinado '{pt_name}' não permitido. Use yolo26n.pt ou yolo26s.pt.",
                )
            arch = pt_name.replace(".pt", "")
            return chosen, f"{arch} (COCO)", pt_name, arch

        raise InvalidMediaException("INVALID_TRAINING_BASE", f"Identificador de base inválido: '{chosen}'.")

    def _enrich_live_metrics(self, record: dict[str, Any]) -> tuple[dict[str, Any], list[TrainingEpochPointDto], list[str]]:
        detect_dir = resolve_repo_path(str(record["detectOutputDir"]))
        results_csv = detect_dir / "results.csv"
        epochs = parse_results_csv(results_csv)
        req_epochs = max(1, int(record.get("epochsRequested") or 1))

        updated = dict(record)
        if epochs:
            last = epochs[-1]
            updated["currentEpoch"] = last.epoch
            updated["epochsCompleted"] = max(int(record.get("epochsCompleted") or 0), len(epochs))
            updated["bestMap50"] = round(max(p.map50 for p in epochs), 4)
            updated["bestMap5095"] = round(max(p.map50_95 for p in epochs), 4)
            if updated["status"] in ("QUEUED", "RUNNING"):
                updated["progressPercent"] = min(99, max(1, int(round((last.epoch / req_epochs) * 100))))
        elif updated["status"] == "COMPLETED":
            updated["progressPercent"] = 100
        elif updated["status"] == "RUNNING":
            updated["progressPercent"] = 1

        if updated["status"] == "COMPLETED":
            updated["progressPercent"] = 100

        log_path = self._run_dir(record["runId"]) / "train.log"
        log_tail: list[str] = []
        if log_path.is_file():
            try:
                all_lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
                log_tail = [ln for ln in all_lines[-40:] if ln.strip()]
            except OSError:
                log_tail = []

        return updated, epochs, log_tail

    def _to_summary_dto(self, rec: dict[str, Any]) -> TrainingRunSummaryDto:
        return TrainingRunSummaryDto.model_validate(rec)

    def get_overview(self) -> TrainingOverviewDto:
        """Return full sequential lineage timeline, available datasets, GPU status, and recent runs."""
        with self._lock:
            records = self._list_raw_records()
            active_id = self._active_run_id

        datasets = self.discover_datasets()
        baseline = self._baseline_metrics()
        next_num, next_label, rec_base_id, rec_base_label = self._next_version_and_recommended_base(records)

        cuda_ok = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None

        v0_fp = next((d.val_fingerprint for d in datasets if d.id == "open-top-yolo-v1"), None)
        lineage_points: list[TrainingLineagePointDto] = [
            TrainingLineagePointDto(
                version_label="v0",
                run_id=None,
                experiment_name=baseline.source_experiment,
                status="BASELINE",
                dataset_id="open-top-yolo-v1",
                val_fingerprint=v0_fp,
                val_dataset_changed=False,
                base_label="yolo26n.pt (COCO)",
                created_at=None,
                epochs_completed=60,
                map50=baseline.map50,
                map50_95=baseline.map50_95,
                precision=baseline.precision,
                recall=baseline.recall,
                best_f1_conf=None,
                registered_model_id=baseline.model_id,
                notes="Baseline PoC inicial (freeze=10 em open-top-yolo-v1).",
            )
        ]

        prev_fp = v0_fp
        enriched_summaries: list[TrainingRunSummaryDto] = []
        for rec in sorted(records, key=lambda r: int(r.get("versionNumber") or 0)):
            enriched, epochs, _ = self._enrich_live_metrics(rec)
            enriched_summaries.append(self._to_summary_dto(enriched))

            status = enriched["status"]
            if status == "COMPLETED" or (status == "RUNNING" and epochs):
                cur_fp = enriched.get("valFingerprint")
                val_changed = bool(prev_fp and cur_fp and prev_fp != cur_fp)
                prev_fp = cur_fp or prev_fp
                last_ep = epochs[-1] if epochs else None
                lineage_points.append(
                    TrainingLineagePointDto(
                        version_label=str(enriched["versionLabel"]),
                        run_id=str(enriched["runId"]),
                        experiment_name=str(enriched["experimentName"]),
                        status=status,
                        dataset_id=str(enriched["datasetId"]),
                        val_fingerprint=cur_fp,
                        val_dataset_changed=val_changed,
                        base_label=str(enriched["baseLabel"]),
                        created_at=enriched.get("createdAt"),
                        epochs_completed=int(enriched.get("epochsCompleted") or 0),
                        map50=enriched.get("finalMap50") if enriched.get("finalMap50") is not None else (last_ep.map50 if last_ep else None),
                        map50_95=enriched.get("finalMap5095")
                        if enriched.get("finalMap5095") is not None
                        else (enriched.get("bestMap5095") if enriched.get("bestMap5095") is not None else (last_ep.map50_95 if last_ep else None)),
                        precision=enriched.get("finalPrecision") if enriched.get("finalPrecision") is not None else (last_ep.precision if last_ep else None),
                        recall=enriched.get("finalRecall") if enriched.get("finalRecall") is not None else (last_ep.recall if last_ep else None),
                        best_f1_conf=enriched.get("bestF1Conf"),
                        registered_model_id=enriched.get("registeredModelId"),
                        notes=enriched.get("notes"),
                    )
                )

        return TrainingOverviewDto(
            baseline_model_id=baseline.model_id,
            baseline_metrics=baseline,
            next_version_number=next_num,
            next_version_label=next_label,
            recommended_base_id=rec_base_id,
            recommended_base_label=rec_base_label,
            active_run_id=active_id,
            gpu_available=cuda_ok,
            gpu_name=gpu_name,
            datasets=datasets,
            lineage_points=lineage_points,
            runs=list(reversed(enriched_summaries)),
            advanced_base_options=self._build_advanced_base_options(rec_base_id, rec_base_label),
        )

    def list_runs(self) -> list[TrainingRunSummaryDto]:
        """List all managed training runs newest first."""
        with self._lock:
            records = self._list_raw_records()
        return [self._to_summary_dto(self._enrich_live_metrics(r)[0]) for r in records]

    def get_run(self, run_id: UUID) -> TrainingRunDetailDto:
        """Return detailed run state, live epoch points, calibration curve, and log tail."""
        rec = self._load_run_record(run_id)
        enriched, epochs, log_tail = self._enrich_live_metrics(rec)
        detect_dir = resolve_repo_path(str(enriched["detectOutputDir"]))
        cal_curve: list[CalibrationCurvePointDto] = []
        summary_file = detect_dir / "eval_summary.json"
        if summary_file.is_file():
            try:
                raw_sum = json.loads(summary_file.read_text(encoding="utf-8"))
                for pt in (raw_sum.get("val_conf_calibration") or {}).get("curve") or []:
                    cal_curve.append(CalibrationCurvePointDto.model_validate(pt))
            except Exception:
                pass
        return TrainingRunDetailDto(
            **self._to_summary_dto(enriched).model_dump(),
            epoch_history=epochs,
            log_tail=log_tail,
            val_calibration_curve=cal_curve,
        )

    def start_run(self, payload: StartTrainingRequest) -> TrainingRunSummaryDto:
        """Start an asynchronous GPU training run, enforcing 1 active training job at a time."""
        if self._require_cuda and not torch.cuda.is_available():
            raise RuntimeUnavailableException(
                "CUDA GPU indisponível neste host; o treinamento gerenciado exige GPU ativa."
            )

        datasets = {d.id: d for d in self.discover_datasets()}
        dataset = datasets.get(payload.dataset_id)
        if dataset is None:
            raise ResourceNotFoundException(
                f"Dataset YOLO '{payload.dataset_id}' não encontrado ou sem classe open_top_freight_wagon."
            )

        with self._lock:
            if self._active_run_id is not None:
                raise ConflictException(
                    f"Já existe um treinamento em execução na GPU ({self._active_run_id}). Aguarde ou pare a execução atual."
                )

            records = self._list_raw_records()
            for r in records:
                if r.get("status") in ("QUEUED", "RUNNING"):
                    raise ConflictException(
                        f"Já existe uma rodada ativa ({r.get('versionLabel')} · {r.get('runId')})."
                    )

            next_num, next_label, _, _ = self._next_version_and_recommended_base(records)
            base_id, base_label, weights_arg, arch = self._resolve_base_weights(payload.base_override, records)

            run_id = uuid4()
            clean_ds = re.sub(r"[^a-zA-Z0-9]+", "_", dataset.id).strip("_")[:24]
            experiment_name = f"train_{next_label}_{clean_ds}_{run_id.hex[:6]}"
            if any(experiment_name.lower().startswith(tok) for tok in RESERVED_BASE_DIR_TOKENS):
                experiment_name = f"ui_{experiment_name}"

            detect_output_dir = self._detect_runs_root / experiment_name
            run_dir = self._run_dir(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"

            record: dict[str, Any] = {
                "runId": str(run_id),
                "versionNumber": next_num,
                "versionLabel": next_label,
                "experimentName": experiment_name,
                "status": "QUEUED",
                "datasetId": dataset.id,
                "dataYamlPath": dataset.data_yaml_path,
                "valFingerprint": dataset.val_fingerprint,
                "baseId": base_id,
                "baseLabel": base_label,
                "architecture": arch,
                "epochsRequested": payload.epochs,
                "epochsCompleted": 0,
                "currentEpoch": 0,
                "progressPercent": 0,
                "imgsz": payload.imgsz,
                "batch": payload.batch,
                "freeze": payload.freeze,
                "lr0": payload.lr0,
                "cosLr": payload.cos_lr,
                "patience": payload.patience,
                "bestMap50": None,
                "bestMap5095": None,
                "finalMap50": None,
                "finalMap5095": None,
                "finalPrecision": None,
                "finalRecall": None,
                "bestF1Conf": None,
                "peakVramMib": None,
                "wallTimeS": None,
                "bestWeightsPath": None,
                "detectOutputDir": _to_repo_rel(detect_output_dir),
                "registeredModelId": None,
                "notes": (payload.notes or "").strip() or None,
                "errorMessage": None,
                "createdAt": _utcnow().isoformat(),
                "startedAt": None,
                "completedAt": None,
            }
            self._save_run_record(record)
            self._active_run_id = run_id

        # Free cached CUDA inference models before spawning subprocess training
        self._runtime_manager.release_cuda_cache()

        cmd = [
            self._python_exe,
            "-u",
            str(self._train_script),
            "--model",
            weights_arg,
            "--data",
            str(resolve_repo_path(dataset.data_yaml_path)),
            "--name",
            experiment_name,
            "--project-dir",
            str(self._detect_runs_root),
            "--epochs",
            str(payload.epochs),
            "--patience",
            str(payload.patience),
            "--imgsz",
            str(payload.imgsz),
            "--batch",
            str(payload.batch),
            "--freeze",
            str(payload.freeze),
            "--lr0",
            str(payload.lr0),
        ]
        if payload.cos_lr:
            cmd.append("--cos-lr")

        thread = threading.Thread(
            target=self._monitor_subprocess,
            args=(run_id, cmd, detect_output_dir, log_path),
            daemon=True,
            name=f"rv-train-{run_id.hex[:8]}",
        )
        with self._lock:
            self._worker_threads = [t for t in self._worker_threads if t.is_alive()]
            self._worker_threads.append(thread)
        thread.start()
        return self._to_summary_dto(record)

    def _monitor_subprocess(
        self,
        run_id: UUID,
        cmd: list[str],
        detect_output_dir: Path,
        log_path: Path,
    ) -> None:
        proc: subprocess.Popen[str] | None = None
        try:
            with self._lock:
                rec = self._load_run_record(run_id)
                rec["status"] = "RUNNING"
                rec["startedAt"] = _utcnow().isoformat()
                self._save_run_record(rec)

            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            with log_path.open("w", encoding="utf-8") as log_handle:
                log_handle.write(f"[RailVision Training] CMD: {' '.join(cmd)}\n")
                log_handle.flush()
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(REPO_ROOT),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    creationflags=creationflags,
                )
                with self._lock:
                    self._active_proc = proc
                return_code = proc.wait()

            with self._lock:
                rec = self._load_run_record(run_id)
                was_stopped = run_id in self._stop_requested or self._shutting_down
                self._stop_requested.discard(run_id)

                epochs = parse_results_csv(detect_output_dir / "results.csv")
                best_pt = detect_output_dir / "weights" / "best.pt"
                last_pt = detect_output_dir / "weights" / "last.pt"
                chosen_pt = best_pt if best_pt.is_file() else (last_pt if last_pt.is_file() else None)
                if chosen_pt is not None:
                    rec["bestWeightsPath"] = _to_repo_rel(chosen_pt)

                if epochs:
                    rec["epochsCompleted"] = len(epochs)
                    rec["currentEpoch"] = epochs[-1].epoch
                    rec["bestMap50"] = round(max(p.map50 for p in epochs), 4)
                    rec["bestMap5095"] = round(max(p.map50_95 for p in epochs), 4)

                summary_path = detect_output_dir / "eval_summary.json"
                if return_code == 0 and not was_stopped and summary_path.is_file():
                    raw_sum = json.loads(summary_path.read_text(encoding="utf-8"))
                    vm = raw_sum.get("val_metrics") or {}
                    cal = raw_sum.get("val_conf_calibration") or {}
                    env = raw_sum.get("environment") or {}
                    hp = raw_sum.get("hyperparameters") or {}
                    rec["status"] = "COMPLETED"
                    rec["progressPercent"] = 100
                    rec["finalMap50"] = float(vm["mAP50"]) if vm.get("mAP50") is not None else rec.get("bestMap50")
                    rec["finalMap5095"] = (
                        float(vm["mAP50_95"]) if vm.get("mAP50_95") is not None else rec.get("bestMap5095")
                    )
                    rec["finalPrecision"] = float(vm["precision"]) if vm.get("precision") is not None else None
                    rec["finalRecall"] = float(vm["recall"]) if vm.get("recall") is not None else None
                    rec["bestF1Conf"] = float(cal["best_f1_conf"]) if cal.get("best_f1_conf") is not None else None
                    rec["peakVramMib"] = (
                        float(env["peak_allocated_vram_mib"]) if env.get("peak_allocated_vram_mib") is not None else None
                    )
                    rec["wallTimeS"] = float(env["train_wall_time_s"]) if env.get("train_wall_time_s") is not None else None
                    if hp.get("epochs_completed") is not None:
                        rec["epochsCompleted"] = int(hp["epochs_completed"])
                elif was_stopped:
                    rec["status"] = "CANCELLED"
                    rec["errorMessage"] = "Treinamento interrompido manualmente pelo usuário."
                else:
                    rec["status"] = "FAILED"
                    tail = ""
                    if log_path.is_file():
                        lines = [ln for ln in log_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
                        tail = " | ".join(lines[-4:])
                    rec["errorMessage"] = f"Processo encerrou com código {return_code}. {tail}".strip()

                rec["completedAt"] = _utcnow().isoformat()
                self._save_run_record(rec)
        except Exception as exc:
            logger.exception("Falha no monitor de treinamento %s", run_id)
            try:
                with self._lock:
                    rec = self._load_run_record(run_id)
                    rec["status"] = "FAILED"
                    rec["errorMessage"] = str(exc)
                    rec["completedAt"] = _utcnow().isoformat()
                    self._save_run_record(rec)
            except Exception:
                pass
        finally:
            with self._lock:
                if self._active_proc is proc:
                    self._active_proc = None
                if self._active_run_id == run_id:
                    self._active_run_id = None

    def _terminate_proc_tree(self, proc: subprocess.Popen[str]) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=4.0)
        except Exception:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    check=False,
                )
            else:
                proc.kill()

    def stop_run(self, run_id: UUID) -> TrainingRunSummaryDto:
        """Stop an active training run and terminate its subprocess tree safely."""
        with self._lock:
            rec = self._load_run_record(run_id)
            if rec.get("status") not in ("QUEUED", "RUNNING"):
                raise ConflictException(
                    f"Execução '{run_id}' não está ativa (estado atual: {rec.get('status')})."
                )
            self._stop_requested.add(run_id)
            proc = self._active_proc if self._active_run_id == run_id else None

        if proc is not None:
            self._terminate_proc_tree(proc)

        # Wait briefly for monitor thread to persist CANCELLED
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            current = self._load_run_record(run_id)
            if current.get("status") not in ("QUEUED", "RUNNING"):
                return self._to_summary_dto(self._enrich_live_metrics(current)[0])
            time.sleep(0.15)

        with self._lock:
            current = self._load_run_record(run_id)
            if current.get("status") in ("QUEUED", "RUNNING"):
                current["status"] = "CANCELLED"
                current["errorMessage"] = "Treinamento interrompido pelo usuário."
                current["completedAt"] = _utcnow().isoformat()
                self._save_run_record(current)
            return self._to_summary_dto(self._enrich_live_metrics(current)[0])

    def register_run_checkpoint(
        self,
        run_id: UUID,
        payload: RegisterTrainingCheckpointRequest | None = None,
    ) -> dict[str, Any]:
        """Register a completed (or best-so-far) training run checkpoint into the Model Registry."""
        req = payload or RegisterTrainingCheckpointRequest()
        rec = self._load_run_record(run_id)
        weights_rel = rec.get("bestWeightsPath")
        if not weights_rel:
            raise ConflictException(
                f"Execução '{rec['versionLabel']}' ainda não gerou checkpoint best.pt para registro."
            )
        weights_path = resolve_repo_path(str(weights_rel))
        if not weights_path.is_file():
            raise ResourceNotFoundException(f"Arquivo de pesos não encontrado: {weights_path}")

        checkpoint_id = req.model_id or f"{rec['experimentName']}_{rec['imgsz']}"
        status = "active" if req.activate else "candidate"
        val_metrics: dict[str, float] = {}
        if rec.get("finalMap50") is not None:
            val_metrics["map50"] = float(rec["finalMap50"])
        elif rec.get("bestMap50") is not None:
            val_metrics["map50"] = float(rec["bestMap50"])
        if rec.get("finalMap5095") is not None:
            val_metrics["map50_95"] = float(rec["finalMap5095"])
        elif rec.get("bestMap5095") is not None:
            val_metrics["map50_95"] = float(rec["bestMap5095"])

        cp = RegisteredCheckpoint(
            id=checkpoint_id,
            model_name=checkpoint_id,
            source_experiment=str(rec["experimentName"]),
            architecture=str(rec.get("architecture") or "yolo26n"),
            status="candidate",
            input_size=int(rec.get("imgsz") or 640),
            class_names={0: "open_top_freight_wagon"},
            pytorch_weights_path=weights_path,
            pytorch_weights_sha256=_sha256_file(weights_path),
            onnx_model_path=None,
            onnx_meta_path=None,
            onnx_model_sha256=None,
            limitations_notice=(
                f"Versão {rec['versionLabel']} treinada em '{rec['datasetId']}' a partir de {rec['baseLabel']}. "
                "Avaliar em val multi-campanha antes de qualquer promoção no Portão G3."
            ),
            val_metrics=val_metrics,
            test_metrics={},
        )
        registered = self._registry.register_checkpoint(cp)
        if status == "active":
            registered = self._registry.set_active_checkpoint(registered.id)

        with self._lock:
            latest = self._load_run_record(run_id)
            latest["registeredModelId"] = registered.id
            self._save_run_record(latest)

        return registered.to_dict()

    def recover_interrupted_runs(self) -> int:
        """Mark QUEUED/RUNNING training runs left behind after an unclean server restart as FAILED."""
        recovered = 0
        with self._lock:
            for rec in self._list_raw_records():
                if rec.get("status") in ("QUEUED", "RUNNING"):
                    rec["status"] = "FAILED"
                    rec["errorMessage"] = "Treinamento interrompido pelo reinício inesperado do backend."
                    rec["completedAt"] = _utcnow().isoformat()
                    self._save_run_record(rec)
                    recovered += 1
        return recovered

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully terminate any active training subprocess on backend shutdown."""
        self._shutting_down = True
        with self._lock:
            proc = self._active_proc
            threads = list(self._worker_threads)
        if proc is not None:
            self._terminate_proc_tree(proc)
        if wait:
            for th in threads:
                th.join(timeout=5.0)
