"""Platform-owned, source-agnostic data model contracts."""

from .catalog import get_data_model_catalog, get_table_definition

__all__ = ["get_data_model_catalog", "get_table_definition"]
