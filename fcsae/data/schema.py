"""Esquema unificado dos exemplos de fact-checking, comum a todos os datasets."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# mapeia rótulos de cada dataset para os canônicos {SUP, REF, NEI}
def norm_label(raw: str) -> str:
    r = (raw or "").strip().upper()
    if "REFUTE" in r or r in ("CONTRADICT", "CONTRADICTION", "FALSE"):
        return "REF"
    if "SUPPORT" in r or r in ("ENTAILMENT", "ENTAIL", "TRUE"):
        return "SUP"
    return "NEI"   # NOT ENOUGH INFO / NEI / NEUTRAL / desconhecido


@dataclass
class Example:
    claim: str
    evidence: List[str] = field(default_factory=list)   # trechos de evidência (texto)
    label: str = "NEI"                                   # SUP | REF | NEI
    source: str = ""                                     # dataset de origem

    def valid(self) -> bool:
        return bool(self.claim and self.claim.strip()) and self.label in ("SUP", "REF", "NEI")
