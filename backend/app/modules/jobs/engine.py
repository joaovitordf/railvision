"""Inspection inference engine contract and DemoFixture parity implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING
from uuid import UUID

import av
import ultralytics

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.modules.vision.runtimes import VisionRuntimeManager
    from app.modules.vision.schemas import DetectionBox


@dataclass(frozen=True, slots=True)
class InferenceRequest:
    """Inputs passed to an inspection inference engine."""

    inspection_id: UUID
    job_id: UUID
    video_path: Path
    content_type: str
    progress_cb: Callable[[int], None] | None = None
    model_id: str | None = None
    runtime: str | None = None
    allow_cpu_fallback: bool = False
    conf_threshold: float | None = None
    iou_threshold: float | None = None
    sample_fps: float | None = None


@dataclass(frozen=True, slots=True)
class ModelVersionDraft:
    """Model version descriptor produced by an inference execution."""

    task: str
    name: str
    version: str
    runtime: str
    weights_sha256: str | None
    artifact_ref: str | None
    demo_placeholder: bool


@dataclass(frozen=True, slots=True)
class ObservationDraft:
    """Single per-frame wagon observation draft."""

    frame_index: int
    frame_timestamp_ms: int
    bbox_x: float
    bbox_y: float
    bbox_width: float
    bbox_height: float
    coordinate_system: str
    detected_class: str
    detection_confidence: float
    material_class: str | None
    material_confidence: float | None
    loading_level: str | None
    loading_confidence: float | None
    example: bool


@dataclass(frozen=True, slots=True)
class ConsolidatedWagonDraft:
    """Consolidated wagon draft matching Java ConsolidatedWagonDraft."""

    track_id: str
    sequence_index: int
    physical_id: str | None
    material_class: str | None
    material_confidence: float | None
    material_rejected_as_unknown: bool
    vote_distribution: dict[str, int]
    loading_level: str
    loading_confidence: float | None
    loading_spread: list[str]
    estimated_fill_percentage: float | None
    measurement_type: str
    physical_volume_verified: bool
    consolidation_method: str | None
    disagreements: list[str]
    quality_flags: list[str]
    review_required: bool
    review_reasons: list[str]
    model_versions: dict[str, str]
    example: bool
    observations: list[ObservationDraft] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class InferenceOutput:
    """Complete output from an inspection inference execution."""

    engine_type: str
    demo_simulation: bool
    used_models: list[ModelVersionDraft]
    wagons: list[ConsolidatedWagonDraft]


class InspectionInferenceEngine(ABC):
    """Abstract inference engine used by asynchronous inspection jobs."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Identifier stored in processing_jobs.engine_type."""

    @property
    @abstractmethod
    def is_demo_simulation(self) -> bool:
        """Whether outputs come from demo fixtures rather than ML weights."""

    @abstractmethod
    def analyze(self, request: InferenceRequest) -> InferenceOutput:
        """Run inspection analysis synchronously in a worker thread."""


class DemoFixtureInferenceEngine(InspectionInferenceEngine):
    """Exact Stage M2 parity port of Java DemoFixtureInferenceEngine."""

    ENGINE_NAME = "DEMO_FIXTURE_ENGINE"
    DEMO_PLACEHOLDER_VERSION = "demo-placeholder-0.0.0"

    def __init__(self, demo_delay_ms: int = 800) -> None:
        self._demo_delay_ms = max(0, int(demo_delay_ms))

    @property
    def engine_name(self) -> str:
        return self.ENGINE_NAME

    @property
    def is_demo_simulation(self) -> bool:
        return True

    def analyze(self, request: InferenceRequest) -> InferenceOutput:
        if self._demo_delay_ms > 0:
            time.sleep(self._demo_delay_ms / 1000.0)

        ver = self.DEMO_PLACEHOLDER_VERSION
        models = [
            ModelVersionDraft("detector", "demo-wagon-detector-placeholder", ver, self.ENGINE_NAME, None, None, True),
            ModelVersionDraft("tracker", "demo-wagon-tracker-placeholder", ver, self.ENGINE_NAME, None, None, True),
            ModelVersionDraft("material", "demo-material-classifier-placeholder", ver, self.ENGINE_NAME, None, None, True),
            ModelVersionDraft("loading", "demo-loading-ordinal-placeholder", ver, self.ENGINE_NAME, None, None, True),
        ]
        model_map = {
            "detector": f"demo-wagon-detector-placeholder@{ver}",
            "tracker": f"demo-wagon-tracker-placeholder@{ver}",
            "material": f"demo-material-classifier-placeholder@{ver}",
            "loading": f"demo-loading-ordinal-placeholder@{ver}",
        }

        wagon1 = ConsolidatedWagonDraft(
            track_id="example-track-001",
            sequence_index=1,
            physical_id=None,
            material_class="iron_ore",
            material_confidence=0.0,
            material_rejected_as_unknown=False,
            vote_distribution={"iron_ore": 2},
            loading_level="high",
            loading_confidence=0.0,
            loading_spread=["high", "high"],
            estimated_fill_percentage=None,
            measurement_type="visual_estimate",
            physical_volume_verified=False,
            consolidation_method="demo_fixture_only",
            disagreements=[],
            quality_flags=["example_fixture"],
            review_required=False,
            review_reasons=[],
            model_versions=model_map,
            example=True,
            observations=[
                ObservationDraft(0, 0, 0.10, 0.24, 0.36, 0.44, "normalized", "freight_wagon_open", 0.0, "iron_ore", 0.0, "high", 0.0, True),
                ObservationDraft(15, 500, 0.28, 0.24, 0.36, 0.44, "normalized", "freight_wagon_open", 0.0, "iron_ore", 0.0, "high", 0.0, True),
            ],
        )

        wagon2 = ConsolidatedWagonDraft(
            track_id="example-track-002",
            sequence_index=2,
            physical_id=None,
            material_class="coal",
            material_confidence=0.0,
            material_rejected_as_unknown=False,
            vote_distribution={"coal": 1, "iron_ore": 1},
            loading_level="medium",
            loading_confidence=0.0,
            loading_spread=["medium", "low"],
            estimated_fill_percentage=None,
            measurement_type="visual_estimate",
            physical_volume_verified=False,
            consolidation_method="demo_fixture_only",
            disagreements=["Observações divergentes entre coal e iron_ore (exemplo fictício)"],
            quality_flags=["example_fixture"],
            review_required=True,
            review_reasons=["observation_disagreement", "low_confidence"],
            model_versions=model_map,
            example=True,
            observations=[
                ObservationDraft(30, 1000, 0.14, 0.25, 0.38, 0.43, "normalized", "freight_wagon_open", 0.0, "coal", 0.0, "medium", 0.0, True),
                ObservationDraft(45, 1500, 0.34, 0.25, 0.38, 0.43, "normalized", "freight_wagon_open", 0.0, "iron_ore", 0.0, "low", 0.0, True),
            ],
        )

        wagon3_empty = ConsolidatedWagonDraft(
            track_id="example-track-003",
            sequence_index=3,
            physical_id=None,
            material_class=None,
            material_confidence=None,
            material_rejected_as_unknown=False,
            vote_distribution={},
            loading_level="empty",
            loading_confidence=0.0,
            loading_spread=["empty"],
            estimated_fill_percentage=None,
            measurement_type="visual_estimate",
            physical_volume_verified=False,
            consolidation_method="demo_fixture_only",
            disagreements=[],
            quality_flags=["example_fixture"],
            review_required=False,
            review_reasons=[],
            model_versions=model_map,
            example=True,
            observations=[
                ObservationDraft(60, 2000, 0.22, 0.26, 0.35, 0.42, "normalized", "freight_wagon_open", 0.0, None, None, "empty", 0.0, True),
            ],
        )

        return InferenceOutput(
            engine_type=self.ENGINE_NAME,
            demo_simulation=True,
            used_models=models,
            wagons=[wagon1, wagon2, wagon3_empty],
        )


class VisionInspectionInferenceEngine(InspectionInferenceEngine):
    """Real GPU/CPU inspection inference engine powered by PyTorch/ONNX + PyAV + ByteTrack."""

    def __init__(
        self,
        *,
        runtime_manager: VisionRuntimeManager,
        settings: Settings,
        sample_fps: float = 5.0,
    ) -> None:
        self._runtime_manager = runtime_manager
        self._settings = settings
        self._sample_fps = max(0.5, float(sample_fps))
        self._runtime_id = runtime_manager.normalize_runtime_id(settings.default_runtime)

    @property
    def engine_name(self) -> str:
        return f"VISION_{self._runtime_id.upper()}"

    @property
    def is_demo_simulation(self) -> bool:
        return False

    def build_active_model_descriptors(self, *, model_id: str | None = None, runtime: str | None = None) -> list[ModelVersionDraft]:
        """Return real non-placeholder model descriptors for detector and tracker."""
        rt_id = self._runtime_manager.normalize_runtime_id(runtime or self._runtime_id)
        meta = self._runtime_manager.resolve_metadata(model_id)
        overview = self._runtime_manager.get_runtime_overview(model_id=model_id)
        display_path, model_sha = self._runtime_manager.model_info_for_runtime(rt_id, model_id=model_id)
        active_provider = str(overview.get("active_execution_provider") or rt_id)
        return [
            ModelVersionDraft(
                task="detector",
                name=meta.model_name,
                version=meta.source_experiment,
                runtime=active_provider,
                weights_sha256=model_sha or None,
                artifact_ref=display_path,
                demo_placeholder=False,
            ),
            ModelVersionDraft(
                task="tracker",
                name="bytetrack",
                version=f"ultralytics-{ultralytics.__version__}",
                runtime="python-ultralytics",
                weights_sha256=None,
                artifact_ref="bytetrack",
                demo_placeholder=False,
            ),
        ]

    def analyze(self, request: InferenceRequest) -> InferenceOutput:
        """Decode video via PyAV, detect wagons, associate ByteTrack IDs, and consolidate honest observations."""
        from app.modules.media.probe import probe_video_metadata
        from app.modules.vision.tracker import ByteTrackSession
        from app.modules.vision.video_pipeline import iter_decoded_video_frames

        metadata = probe_video_metadata(request.video_path)
        sample_fps = max(0.5, float(request.sample_fps)) if request.sample_fps is not None else self._sample_fps
        high_conf = (
            max(0.01, min(0.99, float(request.conf_threshold)))
            if request.conf_threshold is not None
            else self._settings.default_confidence_threshold
        )
        iou_threshold = (
            max(0.05, min(0.95, float(request.iou_threshold)))
            if request.iou_threshold is not None
            else self._settings.default_iou_threshold
        )
        runtime_id = self._runtime_manager.normalize_runtime_id(request.runtime or self._runtime_id)
        tracker = ByteTrackSession(
            high_thresh=high_conf,
            sample_fps=sample_fps,
        )
        detector_conf = tracker.candidate_conf_threshold
        max_frames = max(1, int(self._settings.max_video_frames_per_ui_run))
        min_obs = max(1, int(getattr(self._settings, "min_track_observations", 2)))

        est_sampled = (
            max(1, math.ceil((metadata.duration_ms / 1000.0) * sample_fps))
            if metadata.duration_ms > 0
            else max(1, metadata.estimated_frames)
        )
        est_sampled = min(est_sampled, max_frames)

        grouped_observations: dict[int, list[tuple[int, int, DetectionBox]]] = defaultdict(list)
        ordered_track_ids: list[int] = []
        active_provider: str | None = None

        sample_interval_ms = 1000.0 / sample_fps
        next_sample_ms: float | None = None
        raw_idx = 0
        sampled_idx = 0
        truncated_by_frame_limit = False

        with av.open(str(request.video_path)) as container:
            in_stream = container.streams.video[0]
            for frame in iter_decoded_video_frames(container):
                if frame.pts is not None and frame.time_base is not None:
                    frame_ts_ms = max(0, int(round(float(frame.pts * frame.time_base) * 1000.0)))
                elif in_stream.time_base is not None and frame.pts is not None:
                    frame_ts_ms = max(0, int(round(float(frame.pts * in_stream.time_base) * 1000.0)))
                else:
                    frame_ts_ms = int(round((raw_idx * 1000.0) / max(1.0, metadata.fps)))
                source_frame_idx = raw_idx
                raw_idx += 1

                if next_sample_ms is None:
                    next_sample_ms = float(frame_ts_ms)
                if (frame_ts_ms + 1.0) < next_sample_ms:
                    continue
                if sampled_idx >= max_frames:
                    truncated_by_frame_limit = True
                    break
                while next_sample_ms <= (frame_ts_ms + 1.0):
                    next_sample_ms += sample_interval_ms

                bgr = frame.to_ndarray(format="bgr24")
                det_res = self._runtime_manager.detect_bgr(
                    bgr,
                    conf_threshold=detector_conf,
                    iou_threshold=iou_threshold,
                    runtime=runtime_id,
                    allow_cpu_fallback=request.allow_cpu_fallback,
                    model_id=request.model_id,
                )
                active_provider = det_res.active_execution_provider
                tracked = tracker.update(det_res.detections)
                for box in tracked:
                    if box.track_id is None:
                        continue
                    tid = int(box.track_id)
                    if tid not in grouped_observations:
                        ordered_track_ids.append(tid)
                    grouped_observations[tid].append((source_frame_idx, frame_ts_ms, box))

                sampled_idx += 1
                if request.progress_cb is not None:
                    pct = 15 + int(round((sampled_idx / max(est_sampled, sampled_idx)) * 80.0))
                    request.progress_cb(min(95, max(15, pct)))

        models = self.build_active_model_descriptors(model_id=request.model_id, runtime=runtime_id)
        if active_provider is not None:
            detector_desc = models[0]
            models[0] = ModelVersionDraft(
                task=detector_desc.task,
                name=detector_desc.name,
                version=detector_desc.version,
                runtime=active_provider,
                weights_sha256=detector_desc.weights_sha256,
                artifact_ref=detector_desc.artifact_ref,
                demo_placeholder=False,
            )

        meta = self._runtime_manager.resolve_metadata(request.model_id)
        model_map = {
            "detector": f"{meta.model_name}@{meta.source_experiment}",
            "tracker": f"bytetrack@ultralytics-{ultralytics.__version__}",
        }
        detector_flag = (
            "poc_detector_exp04c"
            if "exp04c" in meta.source_experiment.lower()
            else f"detector_{meta.source_experiment}"
        )
        base_quality_flags = [
            detector_flag,
            "material_not_evaluated",
            "loading_not_evaluated",
        ]
        base_review_reasons = [
            f"{detector_flag}_unapproved",
            "material_classifier_pending",
            "loading_estimator_pending",
        ]
        if truncated_by_frame_limit:
            base_quality_flags.append("truncated_at_max_frames")
            base_review_reasons.append("video_truncated_at_max_frames")

        surviving_track_ids = [
            tid for tid in ordered_track_ids if len(grouped_observations[tid]) >= min_obs
        ]

        wagons: list[ConsolidatedWagonDraft] = []
        for seq_idx, tid in enumerate(surviving_track_ids, start=1):
            entries = grouped_observations[tid]
            obs_drafts = [
                ObservationDraft(
                    frame_index=f_idx,
                    frame_timestamp_ms=ts_ms,
                    bbox_x=round(float(box.norm_x), 6),
                    bbox_y=round(float(box.norm_y), 6),
                    bbox_width=round(float(box.norm_width), 6),
                    bbox_height=round(float(box.norm_height), 6),
                    coordinate_system="normalized",
                    detected_class=box.class_name,
                    detection_confidence=round(float(box.confidence), 4),
                    material_class="not_evaluated",
                    material_confidence=None,
                    loading_level="not_evaluated",
                    loading_confidence=None,
                    example=False,
                )
                for f_idx, ts_ms, box in entries
            ]

            wagons.append(
                ConsolidatedWagonDraft(
                    track_id=f"track-{tid:03d}",
                    sequence_index=seq_idx,
                    physical_id=None,
                    material_class="not_evaluated",
                    material_confidence=None,
                    material_rejected_as_unknown=False,
                    vote_distribution={},
                    loading_level="not_evaluated",
                    loading_confidence=None,
                    loading_spread=[],
                    estimated_fill_percentage=None,
                    measurement_type="not_evaluated",
                    physical_volume_verified=False,
                    consolidation_method="bytetrack_detection_only",
                    disagreements=[],
                    quality_flags=list(base_quality_flags),
                    review_required=True,
                    review_reasons=list(base_review_reasons),
                    model_versions=model_map,
                    example=False,
                    observations=obs_drafts,
                )
            )

        return InferenceOutput(
            engine_type=self.engine_name,
            demo_simulation=False,
            used_models=models,
            wagons=wagons,
        )
