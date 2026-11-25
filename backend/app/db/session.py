"""SQLAlchemy engine and session factory helpers."""

from collections.abc import Generator
from functools import lru_cache
import re
from typing import Any

from fastapi import Request
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings

_SAFE_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def create_db_engine(
    database_url: str | None = None,
    *,
    schema: str | None = None,
    echo: bool = False,
) -> Engine:
    """Create a SQLAlchemy engine for PostgreSQL, optionally pinned to an isolated schema."""
    url = database_url or get_settings().database_url
    connect_args: dict[str, Any] = {"connect_timeout": 2}
    if schema is not None:
        if not _SAFE_SCHEMA_RE.match(schema):
            raise ValueError(f"Invalid PostgreSQL schema identifier: {schema!r}")
        connect_args["options"] = f"-c search_path={schema}"
    return create_engine(url, echo=echo, pool_pre_ping=True, connect_args=connect_args, future=True)


@lru_cache(maxsize=1)
def _default_session_factory() -> sessionmaker[Session]:
    settings: Settings = get_settings()
    engine = create_db_engine(settings.database_url)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session_factory(
    database_url: str | None = None,
    *,
    schema: str | None = None,
) -> sessionmaker[Session]:
    """Return a SQLAlchemy sessionmaker bound to the given URL/schema or cached default settings."""
    if database_url is None and schema is None:
        return _default_session_factory()
    engine = create_db_engine(database_url, schema=schema)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db_session(request: Request) -> Generator[Session, None, None]:
    """FastAPI dependency yielding a scoped SQLAlchemy session from app state or defaults."""
    factory: sessionmaker[Session] | None = getattr(request.app.state, "session_factory", None)
    if factory is None:
        factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()
