from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LEGAL_DIAGNOSTIC_SCHEMA_VERSION = "legality_diagnostics.v1"


@dataclass(frozen=True)
class LegalDiagnostic:
    code: str
    message: str
    path: tuple[str, ...] = ()
    severity: str = "error"
    details: tuple[tuple[str, str], ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "path": list(self.path),
            "severity": self.severity,
            "details": {key: value for key, value in self.details},
        }


@dataclass(frozen=True)
class LegalReport:
    legal: bool
    reasons: tuple[str, ...] = ()
    diagnostics: tuple[LegalDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(str(reason) for reason in self.reasons))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics) or diagnostics_from_reasons(self.reasons))

    @classmethod
    def ok(cls) -> "LegalReport":
        return cls(True, ())

    @classmethod
    def fail(cls, *reasons: str) -> "LegalReport":
        return cls(False, tuple(reasons))

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LEGAL_DIAGNOSTIC_SCHEMA_VERSION,
            "legal": self.legal,
            "reasons": list(self.reasons),
            "diagnostics": [diagnostic.to_json_dict() for diagnostic in self.diagnostics],
        }


def diagnostics_from_reasons(reasons: tuple[str, ...]) -> tuple[LegalDiagnostic, ...]:
    return tuple(_diagnostic_from_reason(reason) for reason in reasons)


def _diagnostic_from_reason(reason: str) -> LegalDiagnostic:
    subject = reason
    path: tuple[str, ...] = ()
    if reason.startswith("team ") and ": " in reason:
        team_prefix, subject = reason.split(": ", 1)
        team_token = team_prefix.split()[1]
        path = ("teams", f"team_{team_token}")
    code = _diagnostic_code(subject)
    if code.startswith("MASK_"):
        path = _mask_path(path)
    elif code == "FORMAT_TEAM_COUNT":
        path = ("format", "n_teams")
    return LegalDiagnostic(
        code=code,
        message=reason,
        path=path,
        severity="error",
        details=(("raw_reason", reason),),
    )


def _diagnostic_code(subject: str) -> str:
    if "duplicate hero" in subject:
        return "DUPLICATE_HERO"
    if "duplicate unique equipment" in subject:
        return "DUPLICATE_UNIQUE_EQUIP"
    if "standing_rank" in subject:
        return "STANDING_ORDER"
    if "unique equipment star" in subject:
        return "UNIQUE_EQUIP_STAR"
    if "team count mismatch" in subject:
        return "FORMAT_TEAM_COUNT"
    if "mask exceeds per-team limit" in subject:
        return "MASK_PER_TEAM_LIMIT"
    if "mask entries" in subject:
        return "MASK_VALUE"
    if "mask exceeds global hidden limit" in subject:
        return "MASK_GLOBAL_LIMIT"
    return "LEGALITY_VIOLATION"


def _mask_path(path: tuple[str, ...]) -> tuple[str, ...]:
    if len(path) == 2 and path[0] == "teams":
        return ("mask", path[1])
    return ("mask",)


_diagnostics_from_reasons = diagnostics_from_reasons
