"""Escrita/leitura de shards de ativações em disco (safetensors).

Cada shard guarda um bloco de token-ativações e metadados por token, para que a
avaliação possa depois computar FVU no geral E especificamente na posição da decisão.
"""
from __future__ import annotations

import os
import json
import glob

import torch
from safetensors.torch import save_file, load_file

LABEL2ID = {"SUP": 0, "REF": 1, "NEI": 2}


class ShardWriter:
    def __init__(self, out_dir: str, shard_tokens: int, meta: dict):
        self.out_dir = out_dir
        self.shard_tokens = shard_tokens
        self.meta = dict(meta)
        os.makedirs(out_dir, exist_ok=True)
        self._acts, self._dec, self._lab, self._src = [], [], [], []
        self._n = 0
        self._shard = 0
        self._total = 0
        self.sources = {}   # nome -> id

    def _src_id(self, name: str) -> int:
        if name not in self.sources:
            self.sources[name] = len(self.sources)
        return self.sources[name]

    def add(self, acts, is_decision, label: str, source: str):
        """acts: [seq, d] fp16 (cpu); is_decision: [seq] bool."""
        seq = acts.shape[0]
        self._acts.append(acts)
        self._dec.append(is_decision.to(torch.uint8))
        self._lab.append(torch.full((seq,), LABEL2ID[label], dtype=torch.uint8))
        self._src.append(torch.full((seq,), self._src_id(source), dtype=torch.uint8))
        self._n += seq
        if self._n >= self.shard_tokens:
            self.flush()

    def flush(self):
        if self._n == 0:
            return
        acts = torch.cat(self._acts, 0)
        tens = {"acts": acts,
                "is_decision": torch.cat(self._dec, 0),
                "label": torch.cat(self._lab, 0),
                "source": torch.cat(self._src, 0)}
        path = os.path.join(self.out_dir, f"shard_{self._shard:05d}.safetensors")
        save_file(tens, path)
        self._total += acts.shape[0]
        self._shard += 1
        self._acts, self._dec, self._lab, self._src = [], [], [], []
        self._n = 0

    def close(self):
        self.flush()
        self.meta.update({"n_shards": self._shard, "n_tokens": self._total,
                          "sources": self.sources, "label2id": LABEL2ID})
        with open(os.path.join(self.out_dir, "manifest.json"), "w") as f:
            json.dump(self.meta, f, indent=1)
        return self.meta


def load_manifest(shard_dir: str) -> dict:
    with open(os.path.join(shard_dir, "manifest.json")) as f:
        return json.load(f)


def iter_shards(shard_dir: str):
    for p in sorted(glob.glob(os.path.join(shard_dir, "shard_*.safetensors"))):
        yield p, load_file(p)
