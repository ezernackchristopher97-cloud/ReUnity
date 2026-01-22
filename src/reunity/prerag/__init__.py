"""
ReUnity Pre-RAG Module

Two-layer filtering system with absurdity gap scoring for
query validation and evidence gating.

DISCLAIMER: This is not a clinical or treatment tool.
"""

from reunity.prerag.absurdity_gap import AbsurdityGapCalculator, AbsurdityGapMetrics
from reunity.prerag.query_gate import QueryGate, QueryGateDecision, QueryGateAction
from reunity.prerag.evidence_gate import EvidenceGate, EvidenceGateDecision, EvidenceGateAction

__all__ = [
    "AbsurdityGapCalculator",
    "AbsurdityGapMetrics",
    "QueryGate",
    "QueryGateDecision",
    "QueryGateAction",
    "EvidenceGate",
    "EvidenceGateDecision",
    "EvidenceGateAction",
]
