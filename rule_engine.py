"""Deterministic routing for ALD requests that must not rely on generation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal


RouteName = Literal["safety", "calculation", "approved_sop", "equipment_lookup", "knowledge_qa"]


@dataclass(frozen=True)
class RouteResult:
    route: RouteName
    rule_id: str
    answer: str | None = None


_UNITS: dict[str, tuple[str, float, str]] = {
    "ms": ("time", 0.001, "s"),
    "s": ("time", 1.0, "s"),
    "sec": ("time", 1.0, "s"),
    "초": ("time", 1.0, "s"),
    "min": ("time", 60.0, "s"),
    "분": ("time", 60.0, "s"),
    "h": ("time", 3600.0, "s"),
    "hour": ("time", 3600.0, "s"),
    "시간": ("time", 3600.0, "s"),
    "pa": ("pressure", 1.0, "Pa"),
    "kpa": ("pressure", 1000.0, "Pa"),
    "torr": ("pressure", 133.322368, "Pa"),
    "mtorr": ("pressure", 0.133322368, "Pa"),
}
_UNIT_PATTERN = "|".join(sorted(map(re.escape, _UNITS), key=len, reverse=True))
_CONVERSION = re.compile(
    rf"(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<source>{_UNIT_PATTERN})"
    rf"\s*(?:를|을|to|->|→)?\s*(?P<target>{_UNIT_PATTERN})"
    rf"(?:로|으로|로 변환|convert)?",
    re.IGNORECASE,
)
_SAFETY_ACTIONS = re.compile(
    r"인터록(?:을|를|이|은|의)?\s*(?:해제|우회)|bypass|override|가스\s*누출|toxic\s*gas|"
    r"배기\s*(?:정지|차단)|위험\s*가스",
    re.IGNORECASE,
)
_SOP_REFERENCE = re.compile(r"\bSOP[-_ ]?[A-Z0-9]+\b|승인된\s*SOP", re.IGNORECASE)
_EQUIPMENT = re.compile(
    r"(?<![A-Z0-9])(?:VG\d+|MFC\s*[-#]?\d+|RF\s*(?:power|generator)|chamber)"
    r"(?![A-Z0-9])|챔버|질량유량계",
    re.IGNORECASE,
)


def _normalise(question: str) -> str:
    return unicodedata.normalize("NFKC", question).strip()


def _convert(question: str) -> RouteResult | None:
    match = _CONVERSION.search(question)
    if not match:
        return None

    source = match.group("source").lower()
    target = match.group("target").lower()
    source_kind, source_scale, _ = _UNITS[source]
    target_kind, target_scale, target_display = _UNITS[target]
    if source_kind != target_kind:
        return RouteResult(
            "calculation",
            "unit.dimension_mismatch",
            "서로 다른 차원의 단위는 변환할 수 없습니다.",
        )

    value = float(match.group("value")) * source_scale / target_scale
    return RouteResult(
        "calculation",
        "unit.convert",
        f"{match.group('value')} {match.group('source')} = {value:.8g} {target_display}",
    )


def route_question(question: str) -> RouteResult:
    """Route risky and deterministic requests before any RAG generation."""
    normalized = _normalise(question)
    if _SAFETY_ACTIONS.search(normalized):
        return RouteResult(
            "safety",
            "safety.escalate",
            "안전·인터록·유해가스 관련 조작은 자동으로 안내하거나 우회할 수 없습니다. "
            "즉시 현장 안전 절차와 승인된 SOP를 확인하고 담당 엔지니어 또는 EHS 책임자에게 "
            "에스컬레이션하세요.",
        )

    conversion = _convert(normalized)
    if conversion:
        return conversion

    if _SOP_REFERENCE.search(normalized):
        return RouteResult(
            "approved_sop",
            "sop.approved_only",
            "승인된 SOP 원문과 버전이 등록되어 있지 않아 절차를 생성할 수 없습니다. "
            "문서 관리자에게 승인된 SOP를 등록하거나 해당 문서 번호를 제공하세요.",
        )

    if _EQUIPMENT.search(normalized):
        return RouteResult("equipment_lookup", "equipment.rag_lookup")
    return RouteResult("knowledge_qa", "knowledge.rag")
