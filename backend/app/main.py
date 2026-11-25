"""FastAPI application factory assembling core middleware, domain routers, and worker lifecycles."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.db.session import get_session_factory
from app.modules.health.router import router as health_router
from app.modules.annotations.router import router as annotations_router
from app.modules.annotations.service import AnnotationStudioService
from app.modules.inspections.router import router as inspections_router
from app.modules.inspections.service import InspectionService
from app.modules.jobs.engine import (
    InspectionInferenceEngine,
    VisionInspectionInferenceEngine,
)
from app.modules.jobs.recovery import recover_interrupted_jobs
from app.modules.jobs.router import router as jobs_router
from app.modules.jobs.service import ProcessingJobService
from app.modules.jobs.worker import JobExecutionManager
from app.modules.lab.router import router as lab_router, vision_lab_router
from app.modules.lab.service import LabService
from app.modules.media.storage import LocalMediaStorageService
from app.modules.media.validator import VideoUploadValidator
from app.modules.results.router import router as models_router
from app.modules.results.service import ResultsService
from app.modules.training.router import router as training_router
from app.modules.training.service import TrainingService
from app.modules.vision.registry import ModelRegistryService
from app.modules.vision.runtimes import VisionRuntimeManager

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    session_factory: sessionmaker[Session] | None = None,
    enable_startup_recovery: bool = True,
    inference_engine: InspectionInferenceEngine | None = None,
) -> FastAPI:
    """Create and configure the full RailVision FastAPI application."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        )

    resolved_settings = settings or get_settings()
    owns_session_factory = session_factory is None
    resolved_session_factory = session_factory or get_session_factory(resolved_settings.database_url)

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
        if enable_startup_recovery:
            try:
                app_instance.state.lab_service.recover_interrupted_runs()
            except Exception as exc:
                logger.warning("Lab startup recovery skipped/failed: %s", exc)
            try:
                app_instance.state.training_service.recover_interrupted_runs()
            except Exception as exc:
                logger.warning("Training startup recovery skipped/failed: %s", exc)
            try:
                recover_interrupted_jobs(app_instance.state.session_factory)
                active_engine = app_instance.state.inference_engine
                if isinstance(active_engine, VisionInspectionInferenceEngine):
                    with app_instance.state.session_factory() as db:
                        if sa.inspect(db.get_bind()).has_table("model_versions"):
                            app_instance.state.results_service.ensure_model_versions(
                                db,
                                active_engine.build_active_model_descriptors(),
                            )
                            db.commit()
            except Exception as exc:
                logger.warning("DB startup recovery/model sync skipped/failed: %s", exc)
        yield
        app_instance.state.worker_manager.shutdown(wait=True)
        app_instance.state.lab_service.shutdown(wait=True)
        app_instance.state.training_service.shutdown(wait=True)
        if owns_session_factory:
            bind = resolved_session_factory.kw.get("bind")
            if bind is not None:
                bind.dispose()

    app = FastAPI(
        title="RailVision AI Backend (Python / FastAPI)",
        version="0.4.0",
        lifespan=lifespan,
    )

    max_upload_bytes = resolved_settings.upload_max_size_mb * 1024 * 1024
    storage_service = LocalMediaStorageService(
        storage_root=resolved_settings.storage_root,
        max_upload_bytes=max_upload_bytes,
    )
    upload_validator = VideoUploadValidator(max_file_size_bytes=max_upload_bytes)
    model_registry = ModelRegistryService(resolved_settings)
    runtime_manager = VisionRuntimeManager(resolved_settings, model_registry=model_registry)
    resolved_engine = inference_engine or VisionInspectionInferenceEngine(
        runtime_manager=runtime_manager,
        settings=resolved_settings,
    )
    results_service = ResultsService()
    worker_manager = JobExecutionManager(
        session_factory=resolved_session_factory,
        storage_service=storage_service,
        inference_engine=resolved_engine,
        results_service=results_service,
        max_concurrency=resolved_settings.processing_max_concurrency,
    )
    job_service = ProcessingJobService(
        inference_engine=resolved_engine,
        worker_manager=worker_manager,
    )
    inspection_service = InspectionService(
        upload_validator=upload_validator,
        storage_service=storage_service,
    )
    lab_service = LabService(settings=resolved_settings, runtime_manager=runtime_manager)
    annotations_service = AnnotationStudioService(settings=resolved_settings, runtime_manager=runtime_manager)
    training_service = TrainingService(
        settings=resolved_settings,
        model_registry=model_registry,
        runtime_manager=runtime_manager,
    )

    app.state.settings = resolved_settings
    app.state.session_factory = resolved_session_factory
    app.state.storage_service = storage_service
    app.state.inference_engine = resolved_engine
    app.state.results_service = results_service
    app.state.worker_manager = worker_manager
    app.state.job_service = job_service
    app.state.inspection_service = inspection_service
    app.state.model_registry = model_registry
    app.state.runtime_manager = runtime_manager
    app.state.lab_service = lab_service
    app.state.annotations_service = annotations_service
    app.state.training_service = training_service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(inspections_router)
    app.include_router(jobs_router)
    app.include_router(models_router)
    app.include_router(vision_lab_router)
    app.include_router(lab_router)
    app.include_router(annotations_router)
    app.include_router(training_router)

    return app
