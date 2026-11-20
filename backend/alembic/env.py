"""Alembic migration environment supporting Flyway coexistence and isolated test schemas."""

from logging.config import fileConfig
from pathlib import Path
import re
import sys
from typing import Any

from alembic import context
import sqlalchemy as sa

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import get_settings
from app.db import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata
_SAFE_SCHEMA_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _get_database_url() -> str:
    configured = config.attributes.get("database_url") or config.get_main_option("sqlalchemy.url")
    if configured:
        return str(configured)
    return get_settings().database_url


def _get_target_schema() -> str | None:
    raw = config.attributes.get("schema")
    if raw is None:
        raw = context.get_x_argument(as_dictionary=True).get("schema")
    if raw is None:
        return None
    schema = str(raw).strip()
    if not _SAFE_SCHEMA_RE.match(schema):
        raise ValueError(f"Invalid PostgreSQL schema identifier: {schema!r}")
    return schema


def include_object(
    object_: Any,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to: Any,
) -> bool:
    """Ignore Flyway's metadata table during autogenerate or metadata comparisons."""
    if type_ == "table" and name == "flyway_schema_history":
        return False
    return True


def run_migrations_offline() -> None:
    """Run migrations in offline SQL generation mode."""
    context.configure(
        url=_get_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    connectable = sa.create_engine(_get_database_url(), poolclass=sa.pool.NullPool, future=True)
    target_schema = _get_target_schema()

    with connectable.connect() as connection:
        configure_kwargs: dict[str, Any] = {
            "connection": connection,
            "target_metadata": target_metadata,
            "include_object": include_object,
            "compare_type": True,
        }
        if target_schema is not None:
            connection.exec_driver_sql(f'SET search_path TO "{target_schema}"')
            connection.commit()
            configure_kwargs["version_table_schema"] = target_schema

        context.configure(**configure_kwargs)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
