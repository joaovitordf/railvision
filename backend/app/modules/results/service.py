"""Results persistence, querying, and DTO mapping service."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import sqlalchemy as sa
from sqlalchemy.orm import Session

from app.core.exceptions import ResourceNotFoundException
from app.db.models import (
    Inspection,
    ModelVersion,
    ProcessingJob,
    WagonObservation,
    WagonResult,
)
from app.modules.jobs.engine import InferenceOutput, ModelVersionDraft
from app.modules.results.schemas import (
    BoundingBoxDto,
    ConsolidationDto,
    LoadingEstimateDto,
    MaterialEstimateDto,
    ModelVersionResponse,
    WagonIdentityDto,
    WagonObservationResponse,
    WagonResultResponse,
)


class ResultsService:
    """Handles persistence and API mapping for model versions, wagons, and observations."""

    def ensure_model_versions(
        self,
        db: Session,
        models: list[ModelVersionDraft],
    ) -> None:
        """Idempotently register or refresh real model metadata rows in model_versions."""
        now = datetime.now(timezone.utc)
        for model in models:
            existing = db.scalar(
                sa.select(ModelVersion).where(
                    ModelVersion.task == model.task,
                    ModelVersion.name == model.name,
                    ModelVersion.version == model.version,
                    ModelVersion.runtime == model.runtime,
                )
            )
            if existing is None:
                db.add(
                    ModelVersion(
                        id=uuid4(),
                        task=model.task,
                        name=model.name,
                        version=model.version,
                        runtime=model.runtime,
                        weights_sha256=model.weights_sha256,
                        artifact_ref=model.artifact_ref,
                        is_demo_placeholder=model.demo_placeholder,
                        created_at=now,
                    )
                )
            elif not existing.is_demo_placeholder and not model.demo_placeholder:
                existing.weights_sha256 = model.weights_sha256
                existing.artifact_ref = model.artifact_ref

    def persist_inference_output(
        self,
        db: Session,
        inspection_id: UUID,
        job_id: UUID,
        output: InferenceOutput,
    ) -> None:
        """Persist used models and consolidated wagon results + observations atomically within a transaction."""
        now = datetime.now(timezone.utc)
        self.ensure_model_versions(db, output.used_models)

        for draft in output.wagons:
            empty_wagon = draft.loading_level.lower() == "empty"
            material_class = None if empty_wagon else draft.material_class
            material_confidence = None if empty_wagon else draft.material_confidence
            rejected_as_unknown = (not empty_wagon) and draft.material_rejected_as_unknown

            wagon_id = uuid4()
            db.add(
                WagonResult(
                    id=wagon_id,
                    inspection_id=inspection_id,
                    job_id=job_id,
                    track_id=draft.track_id,
                    sequence_index=draft.sequence_index,
                    physical_id=draft.physical_id,
                    material_class=material_class,
                    material_confidence=material_confidence,
                    material_rejected_as_unknown=rejected_as_unknown,
                    vote_distribution_json=dict(draft.vote_distribution),
                    loading_level=draft.loading_level,
                    loading_confidence=draft.loading_confidence,
                    loading_spread_json=list(draft.loading_spread),
                    estimated_fill_percentage=draft.estimated_fill_percentage,
                    measurement_type=draft.measurement_type or "visual_estimate",
                    physical_volume_verified=draft.physical_volume_verified,
                    observations_count=len(draft.observations),
                    consolidation_method=draft.consolidation_method,
                    disagreements_json=list(draft.disagreements),
                    quality_flags_json=list(draft.quality_flags),
                    review_required=draft.review_required,
                    review_reasons_json=list(draft.review_reasons),
                    model_versions_json=dict(draft.model_versions),
                    is_example=draft.example,
                    created_at=now,
                )
            )

            for obs in draft.observations:
                db.add(
                    WagonObservation(
                        id=uuid4(),
                        wagon_result_id=wagon_id,
                        frame_index=obs.frame_index,
                        frame_timestamp_ms=obs.frame_timestamp_ms,
                        bbox_x=obs.bbox_x,
                        bbox_y=obs.bbox_y,
                        bbox_width=obs.bbox_width,
                        bbox_height=obs.bbox_height,
                        coordinate_system=obs.coordinate_system or "normalized",
                        detected_class=obs.detected_class,
                        detection_confidence=obs.detection_confidence,
                        material_class=None if empty_wagon else obs.material_class,
                        material_confidence=None if empty_wagon else obs.material_confidence,
                        loading_level=obs.loading_level,
                        loading_confidence=obs.loading_confidence,
                        is_example=obs.example,
                    )
                )

    def _ensure_inspection_exists(self, db: Session, inspection_id: UUID) -> None:
        exists = db.scalar(sa.select(Inspection.id).where(Inspection.id == inspection_id))
        if exists is None:
            raise ResourceNotFoundException(f"Inspeção não encontrada: {inspection_id}")

    def find_latest_completed_job_id(self, db: Session, inspection_id: UUID) -> UUID | None:
        """Return latest COMPLETED processing job UUID for an inspection."""
        return db.scalar(
            sa.select(ProcessingJob.id)
            .where(
                ProcessingJob.inspection_id == inspection_id,
                ProcessingJob.status == "COMPLETED",
            )
            .order_by(ProcessingJob.created_at.desc())
            .limit(1)
        )

    def list_wagons_for_inspection(self, db: Session, inspection_id: UUID) -> list[WagonResult]:
        """List consolidated wagons for the latest completed processing job of an inspection."""
        self._ensure_inspection_exists(db, inspection_id)
        completed_job_id = self.find_latest_completed_job_id(db, inspection_id)
        if completed_job_id is None:
            return []
        return list(
            db.scalars(
                sa.select(WagonResult)
                .where(
                    WagonResult.inspection_id == inspection_id,
                    WagonResult.job_id == completed_job_id,
                )
                .order_by(WagonResult.sequence_index.asc())
            )
        )

    def get_wagon_for_inspection(self, db: Session, inspection_id: UUID, wagon_id: UUID) -> WagonResult:
        """Fetch a single wagon result ensuring ownership by inspection_id."""
        self._ensure_inspection_exists(db, inspection_id)
        wagon = db.get(WagonResult, wagon_id)
        if wagon is None:
            raise ResourceNotFoundException(f"Vagão não encontrado: {wagon_id}")
        if wagon.inspection_id != inspection_id:
            raise ResourceNotFoundException("Vagão não pertence à inspeção informada.")
        return wagon

    def list_observations_for_wagon(
        self,
        db: Session,
        inspection_id: UUID,
        wagon_id: UUID,
    ) -> list[WagonObservation]:
        """List observations for a validated wagon ordered by frame timestamp ascending."""
        self.get_wagon_for_inspection(db, inspection_id, wagon_id)
        return list(
            db.scalars(
                sa.select(WagonObservation)
                .where(WagonObservation.wagon_result_id == wagon_id)
                .order_by(WagonObservation.frame_timestamp_ms.asc())
            )
        )

    def summarize_for_job(self, db: Session, inspection_id: UUID, job_id: UUID | None) -> tuple[int, int]:
        """Return (total_wagons, review_required_count) for a completed job."""
        if job_id is None:
            return 0, 0
        total = db.scalar(
            sa.select(sa.func.count())
            .select_from(WagonResult)
            .where(WagonResult.inspection_id == inspection_id, WagonResult.job_id == job_id)
        )
        review_req = db.scalar(
            sa.select(sa.func.count())
            .select_from(WagonResult)
            .where(
                WagonResult.inspection_id == inspection_id,
                WagonResult.job_id == job_id,
                WagonResult.review_required.is_(True),
            )
        )
        return int(total or 0), int(review_req or 0)

    def list_model_versions(
        self,
        db: Session,
        *,
        include_placeholders: bool = False,
    ) -> list[ModelVersion]:
        """List registered model versions ordered by task ASC, name ASC, created_at DESC."""
        stmt = sa.select(ModelVersion)
        if not include_placeholders:
            stmt = stmt.where(ModelVersion.is_demo_placeholder.is_(False))
        return list(
            db.scalars(
                stmt.order_by(
                    ModelVersion.task.asc(),
                    ModelVersion.name.asc(),
                    ModelVersion.created_at.desc(),
                )
            )
        )

    def to_wagon_response(self, entity: WagonResult) -> WagonResultResponse:
        """Map ORM WagonResult to frontend-compatible camelCase DTO."""
        empty_wagon = entity.loading_level.lower() == "empty"
        material = (
            None
            if empty_wagon
            else MaterialEstimateDto(
                material_class=entity.material_class,
                confidence=entity.material_confidence,
                vote_distribution=dict(entity.vote_distribution_json or {}),
                rejected_as_unknown=entity.material_rejected_as_unknown,
            )
        )
        return WagonResultResponse(
            id=entity.id,
            inspection_id=entity.inspection_id,
            job_id=entity.job_id,
            wagon_identity=WagonIdentityDto(
                track_id=entity.track_id,
                sequence_index=entity.sequence_index,
                physical_id=entity.physical_id,
            ),
            material=material,
            loading_estimate=LoadingEstimateDto(
                level=entity.loading_level,
                level_observation_spread=list(entity.loading_spread_json or []),
                estimated_fill_percentage=entity.estimated_fill_percentage,
                confidence=entity.loading_confidence,
                measurement_type=entity.measurement_type,
                physical_volume_verified=entity.physical_volume_verified,
            ),
            model_versions=dict(entity.model_versions_json or {}),
            consolidation=ConsolidationDto(
                observations_count=entity.observations_count,
                disagreements=list(entity.disagreements_json or []),
                method=entity.consolidation_method,
            ),
            quality_flags=list(entity.quality_flags_json or []),
            review_required=entity.review_required,
            review_reasons=list(entity.review_reasons_json or []),
            is_example=entity.is_example,
            created_at=entity.created_at,
        )

    def to_observation_response(self, entity: WagonObservation) -> WagonObservationResponse:
        """Map ORM WagonObservation to frontend-compatible camelCase DTO."""
        return WagonObservationResponse(
            id=entity.id,
            wagon_result_id=entity.wagon_result_id,
            frame_index=entity.frame_index,
            frame_timestamp_ms=entity.frame_timestamp_ms,
            bounding_box=BoundingBoxDto(
                x=entity.bbox_x,
                y=entity.bbox_y,
                width=entity.bbox_width,
                height=entity.bbox_height,
                coordinate_system=entity.coordinate_system,
            ),
            detected_class=entity.detected_class,
            detection_confidence=entity.detection_confidence,
            material_class=entity.material_class,
            material_confidence=entity.material_confidence,
            loading_level=entity.loading_level,
            loading_confidence=entity.loading_confidence,
            is_example=entity.is_example,
        )

    def to_model_version_response(self, entity: ModelVersion) -> ModelVersionResponse:
        """Map ORM ModelVersion to frontend-compatible camelCase DTO."""
        return ModelVersionResponse(
            id=entity.id,
            task=entity.task,
            name=entity.name,
            version=entity.version,
            runtime=entity.runtime,
            weights_sha256=entity.weights_sha256,
            artifact_ref=entity.artifact_ref,
            is_demo_placeholder=entity.is_demo_placeholder,
            created_at=entity.created_at,
        )
