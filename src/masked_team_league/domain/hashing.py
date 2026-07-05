from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
from typing import Any, TypeAlias

Slot: TypeAlias = tuple[int, int]


def _canonical(obj: Any) -> Any:
    if is_dataclass(obj):
        return {item.name: _canonical(getattr(obj, item.name)) for item in fields(obj)}
    if isinstance(obj, dict):
        return {str(key): _canonical(value) for key, value in sorted(obj.items(), key=lambda item: str(item[0]))}
    if isinstance(obj, (tuple, list)):
        return [_canonical(value) for value in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_canonical(value) for value in obj)
    return obj


def canonical_hash(obj: Any) -> str:
    payload = json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cached_canonical_hash(obj: Any, attr_name: str = "_canonical_hash_cache") -> str:
    cached = getattr(obj, attr_name, None)
    if cached is not None:
        return str(cached)
    value = canonical_hash(obj)
    object.__setattr__(obj, attr_name, value)
    return value


_cached_canonical_hash = cached_canonical_hash
