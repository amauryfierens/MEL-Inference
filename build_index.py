"""Encode every name of a concept base (canonical name and aliases, each (concept, name) once) with the retriever.

    python build_index.py --kb data/kb_fr.jsonl --out data/index_bioelx_fr [--encoder bioelx-nlp/bioelx_retriever]

Writes vectors.npy (float16), cuis.json and DONE. The French base has several million names: a few hours on one GPU,
about 2 bytes x 768 per name on disk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from retriever import Encoder, read_kb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--encoder", default="bioelx-nlp/bioelx_retriever")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args()
    from tqdm import tqdm

    names, cuis = [], []
    for c in read_kb(a.kb):
        for n in dict.fromkeys([c["canonical_name"], *c["aliases"]]):
            names.append(n)
            cuis.append(c["concept_id"])
    print(f"{len(set(cuis)):,} concepts, {len(names):,} names", flush=True)

    enc = Encoder(a.encoder, a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    vec = np.lib.format.open_memmap(a.out / "vectors.tmp.npy", mode="w+", dtype=np.float16, shape=(len(names), enc.dimension))
    step = a.batch_size * 100
    for s in tqdm(range(0, len(names), step), desc="encoding", unit="block"):
        vec[s:s + step] = enc.encode(names[s:s + step], a.batch_size).astype(np.float16)
    vec.flush()
    del vec
    (a.out / "vectors.tmp.npy").rename(a.out / "vectors.npy")
    (a.out / "cuis.json").write_text(json.dumps(cuis))
    (a.out / "manifest.json").write_text(json.dumps({"encoder": a.encoder, "kb": str(a.kb), "names": len(names)}))
    (a.out / "DONE").touch()
    print(f"index -> {a.out}")


if __name__ == "__main__":
    main()
