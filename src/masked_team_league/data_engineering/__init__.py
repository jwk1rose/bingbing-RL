"""数据工程层，对应 tex §1380-1415。

这里放 core tables、run metadata、artifact hash 和可复现性清单。
"""

from .core_tables import (
    CORE_TABLE_SCHEMA_VERSION,
    LeagueStrategyTableRecord,
    LoadoutTableRecord,
    ObservationTableRecord,
    PlanMatchTableRecord,
    SingleMatchupTableRecord,
    load_table_jsonl,
    write_table_jsonl,
)
from .run_metadata import (
    RUN_METADATA_SCHEMA_VERSION,
    RunArtifactRef,
    RunMetadataManifest,
    hash_generation_config,
    load_run_metadata_manifest,
    write_run_metadata_manifest,
)

__all__ = [
    "CORE_TABLE_SCHEMA_VERSION",
    "LeagueStrategyTableRecord",
    "LoadoutTableRecord",
    "ObservationTableRecord",
    "PlanMatchTableRecord",
    "RUN_METADATA_SCHEMA_VERSION",
    "RunArtifactRef",
    "RunMetadataManifest",
    "SingleMatchupTableRecord",
    "hash_generation_config",
    "load_run_metadata_manifest",
    "load_table_jsonl",
    "write_run_metadata_manifest",
    "write_table_jsonl",
]
