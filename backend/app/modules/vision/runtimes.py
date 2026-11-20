"""Computer vision inference runtimes (PyTorch CUDA default, PyTorch CPU, selectable ONNX CPU)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

from app.core.config import Settings
from app.core.exceptions import InvalidMediaException, RuntimeUnavailableException
from app.core.paths import REPO_ROOT
from app.modules.vision.preprocess import decode_yolo_output, preprocess_bgr_letterbox
from app.modules.vision.schemas import FrameDetectionResult, RuntimeOptionDto

SUPPORTED_RUNTIMES = ("pytorch_cuda", "pytorch_cpu", "onnx_cpu")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


@dataclass(frozen=True, slots=True)
class ModelArtifactMetadata:
    """Loaded metadata sidecar and file checksums for the active detection checkpoint."""

    model_name: str
    source_experiment: str
    input_size: int
    input_shape: list[int]
    output_shape: list[int]
    class_names: dict[int, str]
    limitations_notice: str
    pytorch_weights_path: Path
    pytorch_weights_sha256: str | None
    onnx_model_path: Path
    onnx_model_sha256: str | None


class VisionRuntimeManager:
    """Manages explicit PyTorch CUDA, PyTorch CPU, and ONNX Runtime CPU inference execution."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._pt_cuda_model: torch.nn.Module | None = None
        self._pt_cpu_model: torch.nn.Module | None = None
        self._ort_session: ort.InferenceSession | None = None
        self._ort_input_name: str | None = None
        self._ort_output_name: str | None = None
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> ModelArtifactMetadata:
        meta_path = self._settings.onnx_model_metadata_path
        pt_path = self._settings.pytorch_weights_path
        onnx_path = self._settings.onnx_model_path

        raw: dict[str, Any] = {}
        if meta_path.is_file():
            raw = json.loads(meta_path.read_text(encoding="utf-8"))

        raw_classes = raw.get("classNames") or {"0": "open_top_freight_wagon"}
        class_names = {int(k): str(v) for k, v in raw_classes.items()}

        pt_sha = _sha256_file(pt_path) if pt_path.is_file() else None
        onnx_sha = str(raw.get("onnxSha256")) if raw.get("onnxSha256") else (
            _sha256_file(onnx_path) if onnx_path.is_file() else None
        )

        return ModelArtifactMetadata(
            model_name=str(raw.get("modelName") or "exp04c_yolo26n_opentop_640"),
            source_experiment=str(raw.get("sourceExperiment") or "exp04c_yolo26n_opentop_freeze10"),
            input_size=int(raw.get("imgsz") or 640),
            input_shape=[int(x) for x in (raw.get("inputShape") or [1, 3, 640, 640])],
            output_shape=[int(x) for x in (raw.get("outputShape") or [1, 5, 8400])],
            class_names=class_names,
            limitations_notice=str(
                raw.get("limitationsNotice")
                or "PoC model (exp04c): bom desempenho em validação, generalização limitada no teste. Não aprovado para produção."
            ),
            pytorch_weights_path=pt_path,
            pytorch_weights_sha256=pt_sha,
            onnx_model_path=onnx_path,
            onnx_model_sha256=onnx_sha,
        )

    @property
    def metadata(self) -> ModelArtifactMetadata:
        """Return loaded checkpoint/ONNX metadata."""
        return self._metadata

    def normalize_runtime_id(self, runtime: str | None) -> str:
        """Validate requested runtime identifier against supported options."""
        resolved = (runtime or self._settings.default_runtime or "pytorch_cuda").strip().lower()
        if resolved not in SUPPORTED_RUNTIMES:
            raise InvalidMediaException(
                "INVALID_RUNTIME",
                f"Runtime inválido '{runtime}'. Valores suportados: {', '.join(SUPPORTED_RUNTIMES)}.",
            )
        return resolved

    def _get_pytorch_model(self, device: str) -> torch.nn.Module:
        weights_path = self._metadata.pytorch_weights_path
        if not weights_path.is_file():
            raise RuntimeUnavailableException(
                f"Checkpoint PyTorch não encontrado em {_display_repo_path(weights_path)}."
            )
        with self._lock:
            if device == "cuda:0":
                if self._pt_cuda_model is None:
                    model = YOLO(str(weights_path)).model.eval().to("cuda:0")
                    dummy = torch.zeros(
                        (1, 3, self._metadata.input_size, self._metadata.input_size),
                        dtype=torch.float32,
                        device="cuda:0",
                    )
                    with torch.inference_mode():
                        _ = model(dummy)
                    torch.cuda.synchronize()
                    self._pt_cuda_model = model
                return self._pt_cuda_model

            if self._pt_cpu_model is None:
                model = YOLO(str(weights_path)).model.eval().cpu()
                self._pt_cpu_model = model
            return self._pt_cpu_model

    def _get_onnx_session(self) -> tuple[ort.InferenceSession, str, str]:
        onnx_path = self._metadata.onnx_model_path
        if not onnx_path.is_file():
            raise RuntimeUnavailableException(
                f"Modelo ONNX não encontrado em {_display_repo_path(onnx_path)}."
            )
        with self._lock:
            if self._ort_session is None:
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(
                    str(onnx_path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._ort_session = sess
                self._ort_input_name = sess.get_inputs()[0].name
                self._ort_output_name = sess.get_outputs()[0].name
            return self._ort_session, str(self._ort_input_name), str(self._ort_output_name)

    def model_info_for_runtime(self, runtime_id: str) -> tuple[str, str]:
        """Return (display_path, sha256) matching the executed model format."""
        if runtime_id == "onnx_cpu":
            return (
                _display_repo_path(self._metadata.onnx_model_path),
                self._metadata.onnx_model_sha256 or "",
            )
        return (
            _display_repo_path(self._metadata.pytorch_weights_path),
            self._metadata.pytorch_weights_sha256 or "",
        )

    def list_runtime_options(self) -> list[RuntimeOptionDto]:
        """Describe all selectable runtimes and explicit live hardware availability."""
        cuda_ok = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
        pt_exists = self._metadata.pytorch_weights_path.is_file()
        onnx_exists = self._metadata.onnx_model_path.is_file()
        ort_providers = list(ort.get_available_providers())

        return [
            RuntimeOptionDto(
                id="pytorch_cuda",
                label="PyTorch + Ultralytics CUDA (Padrão)",
                device_type="GPU" if cuda_ok else "UNAVAILABLE",
                available=cuda_ok and pt_exists,
                provider=f"PyTorch CUDA ({gpu_name})" if cuda_ok else "CUDAExecutionUnavailable",
                diagnostic=(
                    f"CUDA ativo em GPU local ({gpu_name}) com PyTorch {torch.__version__}."
                    if cuda_ok
                    else "CUDA não disponível no dispositivo local (sem fallback silencioso)."
                ),
            ),
            RuntimeOptionDto(
                id="pytorch_cpu",
                label="PyTorch + Ultralytics CPU",
                device_type="CPU",
                available=pt_exists,
                provider="PyTorch CPU",
                diagnostic=f"Execução explícita em CPU via PyTorch {torch.__version__}.",
            ),
            RuntimeOptionDto(
                id="onnx_cpu",
                label="ONNX Runtime CPU",
                device_type="CPU",
                available=onnx_exists and "CPUExecutionProvider" in ort_providers,
                provider="CPUExecutionProvider",
                diagnostic=f"ONNX Runtime {ort.__version__} em CPUExecutionProvider.",
            ),
        ]

    def get_runtime_overview(self) -> dict[str, Any]:
        """Return detailed runtime status payload preserving Java contract plus Python options."""
        cuda_ok = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else None
        pt_exists = self._metadata.pytorch_weights_path.is_file()
        onnx_exists = self._metadata.onnx_model_path.is_file()
        default_rt = self.normalize_runtime_id(self._settings.default_runtime)

        available = False
        unavailable_reason: str | None = None
        device_type = "CPU"
        active_provider = "CPUExecutionProvider"
        device_diagnostic = ""

        if default_rt == "pytorch_cuda":
            available = cuda_ok and pt_exists
            if available:
                device_type = "GPU"
                active_provider = f"PyTorch CUDA ({gpu_name})"
                device_diagnostic = (
                    f"Runtime padrão PyTorch {torch.__version__} ativo em GPU ({gpu_name}); "
                    "ONNX Runtime CPU e PyTorch CPU selecionáveis por execução."
                )
            else:
                device_type = "UNAVAILABLE"
                active_provider = "CUDAUnavailable"
                unavailable_reason = (
                    "Checkpoint PyTorch ausente."
                    if not pt_exists
                    else "CUDA indisponível neste host para o runtime padrão pytorch_cuda."
                )
                device_diagnostic = unavailable_reason
        elif default_rt == "pytorch_cpu":
            available = pt_exists
            device_type = "CPU"
            active_provider = "PyTorch CPU"
            device_diagnostic = f"Runtime padrão PyTorch CPU ({torch.__version__})."
            if not available:
                unavailable_reason = "Checkpoint PyTorch não encontrado."
        else:
            available = onnx_exists
            device_type = "CPU"
            active_provider = "CPUExecutionProvider"
            device_diagnostic = f"ONNX Runtime {ort.__version__} em CPUExecutionProvider."
            if not available:
                unavailable_reason = "Modelo ONNX não encontrado."

        display_path, sha = self.model_info_for_runtime(default_rt)
        return {
            "available": available,
            "unavailable_reason": unavailable_reason,
            "model_name": self._metadata.model_name,
            "source_experiment": self._metadata.source_experiment,
            "model_display_path": display_path,
            "model_sha256": sha,
            "input_size": self._metadata.input_size,
            "input_shape": self._metadata.input_shape,
            "output_shape": self._metadata.output_shape,
            "class_names": self._metadata.class_names,
            "requested_execution_provider": default_rt,
            "active_execution_provider": active_provider,
            "device_type": device_type,
            "cuda_available": cuda_ok,
            "fallback_occurred": False,
            "available_providers": list(ort.get_available_providers()),
            "device_diagnostic": device_diagnostic,
            "limitations_notice": self._metadata.limitations_notice,
            "runtime_options": self.list_runtime_options(),
        }

    def detect_bgr(
        self,
        bgr_frame: np.ndarray,
        *,
        conf_threshold: float,
        iou_threshold: float,
        runtime: str | None = None,
        allow_cpu_fallback: bool = False,
    ) -> FrameDetectionResult:
        """Run deterministic preprocessing, model forward pass, and YOLO decode with explicit provenance."""
        requested = self.normalize_runtime_id(runtime)
        t_wall0 = time.perf_counter()

        t_pre0 = time.perf_counter()
        nchw, lb_meta = preprocess_bgr_letterbox(bgr_frame, target_size=self._metadata.input_size)
        preprocess_ms = (time.perf_counter() - t_pre0) * 1000.0

        fallback_occurred = False
        effective_runtime = requested
        device_type: str
        active_provider: str

        if requested == "pytorch_cuda":
            if torch.cuda.is_available():
                model = self._get_pytorch_model("cuda:0")
                device_type = "GPU"
                active_provider = f"PyTorch CUDA ({torch.cuda.get_device_name(0)})"
                t_inf0 = time.perf_counter()
                with torch.inference_mode():
                    tensor = torch.from_numpy(nchw).to("cuda:0")
                    out = model(tensor)
                    raw_tensor = out[0] if isinstance(out, (tuple, list)) else out
                    torch.cuda.synchronize()
                    raw_np = raw_tensor.detach().cpu().numpy()
                inference_ms = (time.perf_counter() - t_inf0) * 1000.0
            else:
                if not allow_cpu_fallback:
                    raise RuntimeUnavailableException(
                        "CUDA não está disponível neste ambiente para o runtime pytorch_cuda. "
                        "Selecione pytorch_cpu ou onnx_cpu, ou habilite explicitamente allowCpuFallback=true."
                    )
                fallback_occurred = True
                effective_runtime = "pytorch_cpu"
                model = self._get_pytorch_model("cpu")
                device_type = "CPU"
                active_provider = "PyTorch CPU (fallback explícito)"
                t_inf0 = time.perf_counter()
                with torch.inference_mode():
                    out = model(torch.from_numpy(nchw).cpu())
                    raw_tensor = out[0] if isinstance(out, (tuple, list)) else out
                    raw_np = raw_tensor.detach().cpu().numpy()
                inference_ms = (time.perf_counter() - t_inf0) * 1000.0
        elif requested == "pytorch_cpu":
            model = self._get_pytorch_model("cpu")
            device_type = "CPU"
            active_provider = "PyTorch CPU"
            t_inf0 = time.perf_counter()
            with torch.inference_mode():
                out = model(torch.from_numpy(nchw).cpu())
                raw_tensor = out[0] if isinstance(out, (tuple, list)) else out
                raw_np = raw_tensor.detach().cpu().numpy()
            inference_ms = (time.perf_counter() - t_inf0) * 1000.0
        else:
            sess, input_name, output_name = self._get_onnx_session()
            device_type = "CPU"
            active_provider = "CPUExecutionProvider"
            t_inf0 = time.perf_counter()
            raw_np = np.asarray(sess.run([output_name], {input_name: nchw})[0])
            inference_ms = (time.perf_counter() - t_inf0) * 1000.0

        t_post0 = time.perf_counter()
        detections = decode_yolo_output(
            raw_np,
            lb_meta,
            self._metadata.class_names,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )
        postprocess_ms = (time.perf_counter() - t_post0) * 1000.0
        total_ms = (time.perf_counter() - t_wall0) * 1000.0

        return FrameDetectionResult(
            detections=detections,
            preprocess_ms=preprocess_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            total_ms=total_ms,
            runtime_id=effective_runtime,
            requested_execution_provider=requested,
            active_execution_provider=active_provider,
            device_type=device_type,
            fallback_occurred=fallback_occurred,
        )
