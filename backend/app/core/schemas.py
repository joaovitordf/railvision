"""Shared Pydantic base models ensuring camelCase JSON compatibility with the Angular frontend."""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """Base schema serializing snake_case Python attributes as camelCase JSON fields."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        protected_namespaces=(),
    )
