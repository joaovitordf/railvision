"""RFC 7807 / 9457 ProblemDetail exception hierarchy and FastAPI exception handlers."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)
PROBLEM_JSON_MEDIA_TYPE = "application/problem+json"


class ProblemException(Exception):
    """Base domain exception serialized as application/problem+json."""

    def __init__(
        self,
        *,
        status_code: int,
        title: str,
        detail: str,
        error_code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.title = title
        self.detail = detail
        self.error_code = error_code
        self.headers = headers or {}


class InvalidMediaException(ProblemException):
    """Raised when uploaded media fails format, signature, or parameter validation."""

    def __init__(self, error_code: str, detail: str) -> None:
        super().__init__(
            status_code=400,
            title="Mídia inválida",
            detail=detail,
            error_code=error_code,
        )


class MediaTooLargeException(ProblemException):
    """Raised when an uploaded file exceeds the configured server size limit."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            status_code=413,
            title="Arquivo excede o limite",
            detail=detail,
        )


class ResourceNotFoundException(ProblemException):
    """Raised when a requested database or run resource does not exist."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            status_code=404,
            title="Recurso não encontrado",
            detail=detail,
        )


class MediaNotFoundException(ProblemException):
    """Raised when an expected local media file cannot be read on disk."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            status_code=404,
            title="Recurso não encontrado",
            detail=detail,
        )


class ConflictException(ProblemException):
    """Raised on concurrent state conflicts (e.g. duplicate active processing job)."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            status_code=409,
            title="Conflito de processamento",
            detail=detail,
        )


class RuntimeUnavailableException(ProblemException):
    """Raised when a requested inference runtime or hardware device is unavailable."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            status_code=503,
            title="Runtime de inferência indisponível",
            detail=detail,
        )


class InvalidMediaRangeException(ProblemException):
    """Raised when an HTTP Range request falls outside valid byte bounds."""

    def __init__(self, content_length: int, detail: str = "Intervalo de bytes inválido.") -> None:
        super().__init__(
            status_code=416,
            title="Intervalo de bytes inválido",
            detail=detail,
            headers={"Content-Range": f"bytes */{content_length}"},
        )
        self.content_length = content_length


def _build_problem_payload(
    *,
    request: Request,
    status_code: int,
    title: str,
    detail: str,
    error_code: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "type": "about:blank",
        "title": title,
        "status": status_code,
        "detail": detail,
        "instance": request.url.path,
    }
    if error_code is not None:
        payload["errorCode"] = error_code
    return payload


def register_exception_handlers(app: FastAPI) -> None:
    """Attach RFC 9457 ProblemDetail exception handlers matching the Angular client expectations."""

    @app.exception_handler(ProblemException)
    def handle_problem_exception(request: Request, exc: ProblemException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            media_type=PROBLEM_JSON_MEDIA_TYPE,
            headers=exc.headers,
            content=_build_problem_payload(
                request=request,
                status_code=exc.status_code,
                title=exc.title,
                detail=exc.detail,
                error_code=exc.error_code,
            ),
        )

    @app.exception_handler(RequestValidationError)
    def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(part) for part in first.get("loc", ()))
        msg = first.get("msg", "Requisição inválida.")
        detail = f"{loc}: {msg}" if loc else str(msg)
        return JSONResponse(
            status_code=400,
            media_type=PROBLEM_JSON_MEDIA_TYPE,
            content=_build_problem_payload(
                request=request,
                status_code=400,
                title="Requisição inválida",
                detail=detail,
                error_code="VALIDATION_ERROR",
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            media_type=PROBLEM_JSON_MEDIA_TYPE,
            headers=getattr(exc, "headers", None),
            content=_build_problem_payload(
                request=request,
                status_code=exc.status_code,
                title="Erro HTTP",
                detail=str(exc.detail),
            ),
        )

    @app.exception_handler(Exception)
    def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception at %s", request.url.path, exc_info=exc)
        return JSONResponse(
            status_code=500,
            media_type=PROBLEM_JSON_MEDIA_TYPE,
            content=_build_problem_payload(
                request=request,
                status_code=500,
                title="Erro interno",
                detail="Ocorreu um erro interno ao processar a requisição.",
            ),
        )
