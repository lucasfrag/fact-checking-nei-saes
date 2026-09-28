"""Loaders dos datasets de fact-checking -> stream de `Example` unificados.

Só datasets com o TEXTO da evidência inline (verificado por sondagem):
- vitaminc  (tals/vitaminc)               : claim, evidence(texto), label
- fever     (copenlu/fever_gold_evidence) : claim, evidence(lista [page,id,texto]), label
Novos loaders (AVeriTeC, SciFact) entram aqui seguindo o mesmo contrato.
"""
from __future__ import annotations

from itertools import islice
from typing import Iterator

from datasets import load_dataset

from .schema import Example, norm_label


def _extract_fever_evidence(ev) -> list:
    """FEVER copenlu: evidence é lista (possivelmente aninhada) de [page, sent_id, texto]."""
    out = []
    def walk(x):
        if isinstance(x, (list, tuple)):
            if len(x) >= 3 and isinstance(x[2], str) and len(x[2]) > 3:
                out.append(x[2].strip())
            else:
                for e in x:
                    walk(e)
    walk(ev)
    # dedup preservando ordem
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t); uniq.append(t)
    return uniq


def load_vitaminc(max_n: int = 0, shuffle: bool = True) -> Iterator[Example]:
    ds = load_dataset("tals/vitaminc", split="train", streaming=True)
    if shuffle:
        ds = ds.shuffle(seed=0, buffer_size=10_000)
    it = islice(ds, max_n) if max_n else ds
    for r in it:
        ev = r.get("evidence")
        ev = [ev.strip()] if isinstance(ev, str) and ev.strip() else []
        yield Example(claim=(r.get("claim") or "").strip(), evidence=ev,
                      label=norm_label(r.get("label")), source="vitaminc")


def load_fever(max_n: int = 0, shuffle: bool = True) -> Iterator[Example]:
    ds = load_dataset("copenlu/fever_gold_evidence", split="train", streaming=True)
    if shuffle:
        ds = ds.shuffle(seed=0, buffer_size=10_000)
    it = islice(ds, max_n) if max_n else ds
    for r in it:
        yield Example(claim=(r.get("claim") or "").strip(),
                      evidence=_extract_fever_evidence(r.get("evidence")),
                      label=norm_label(r.get("label")), source="fever")


def iter_general(dataset: str, max_n: int = 0) -> Iterator[str]:
    """Texto geral (ex.: the Pile) para dar ao SAE features gerais além das de domínio.
    Rende strings de texto cru (sem template de veredito)."""
    ds = load_dataset(dataset, split="train", streaming=True).shuffle(seed=0, buffer_size=10_000)
    it = islice(ds, max_n) if max_n else ds
    for r in it:
        t = (r.get("text") or "").strip()
        if len(t) > 200:                       # descarta trechos muito curtos
            yield t


REGISTRY = {"vitaminc": load_vitaminc, "fever": load_fever}


def iter_examples(names, max_per_dataset: int = 0, shuffle: bool = True) -> Iterator[Example]:
    for name in names:
        if name not in REGISTRY:
            raise KeyError(f"loader desconhecido: {name} (disponíveis: {list(REGISTRY)})")
        for ex in REGISTRY[name](max_n=max_per_dataset, shuffle=shuffle):
            if ex.valid():
                yield ex
