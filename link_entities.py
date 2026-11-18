"""Step 2: link the entities found by Report-Extractor to UMLS concepts.

    python link_entities.py ENTITIES.json --report REPORT.txt --out outputs/report   [--no-rerank] [--top 5]

ENTITIES.json is the output of `extract_medical_entities.py --json` (text, label, score, start, end).
1. Retriever: each distinct mention text gets the top 100 concepts of the index (retriever.py).
2. Cross-encoder (xMEN's published model for the language, e.g. phlobo/xmen-fr-ce-medmentions): reads each mention
   in its context, "<128 characters> [START] mention [END] <128 characters>", against each of the retriever's first
   --rerank-k candidates written as xMEN writes them, "TUIs [TYPE] canonical name [TITLE] alias [SEP] alias ...",
   and orders them by its score (the other candidates keep the retriever's order after them).
The final concept is the cross-encoder's first, or the retriever's first with --no-rerank; both orders are written.

Writes OUT.json (one record per entity, with the top --top of both orders) and OUT.tsv (one line per entity).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch

from retriever import Encoder, Retriever

NOT_LINKED = ["Dosage"]          # a quantity, not a concept
CONTEXT = 128                    # characters on each side, as xMEN


def candidate_text(c: dict) -> str:
    """As xMEN writes a candidate (xmen.reranking.cross_encoder._candidate_text, with semantic types)."""
    names = [c["canonical_name"], *c["aliases"]]
    return ",".join(c["types"]) + " [TYPE] " + names[0] + " [TITLE] " + " [SEP] ".join(names[1:])


def marked(text: str, start: int, end: int, n: int = CONTEXT) -> str:
    flat = lambda s: " ".join(s.split())
    return f"{flat(text[max(0, start - n):start])} [START] {text[start:end]} [END] {flat(text[end:end + n])}"


def read_concepts(kb: Path, need: set[str]) -> dict[str, dict]:
    out = {}
    with Path(kb).open(encoding="utf-8") as f:
        for line in f:
            if line[16:24] in need:            # '{"concept_id": "C1234567"...': skip the JSON parse of the rest
                c = json.loads(line)
                out[c["concept_id"]] = c
    return out


class CrossEncoder:
    def __init__(self, name: str, device: str = "cuda", max_length: int = 512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device).eval()
        self.max_length = max_length

    @torch.inference_mode()
    def __call__(self, pairs: list[tuple[str, str]], batch_size: int = 64) -> list[float]:
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        scores = [0.0] * len(pairs)
        auto = torch.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else torch.autocast("cpu", enabled=False)
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            tok = self.tokenizer([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], padding=True,
                                 truncation="longest_first", max_length=self.max_length, return_tensors="pt").to(self.device)
            with auto:
                logits = self.model(**tok).logits[:, 0].float().cpu().tolist()
            for i, v in zip(idx, logits):
                scores[i] = v
        return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entities", type=Path)
    ap.add_argument("--report", type=Path, required=True, help="the report the entities come from (for the context)")
    ap.add_argument("--out", type=Path, required=True, help="prefix: writes OUT.json and OUT.tsv")
    ap.add_argument("--no-rerank", action="store_true", help="retriever only")
    ap.add_argument("--rerank-k", type=int, default=64, help="candidates read by the cross-encoder (xMEN: 64)")
    ap.add_argument("--skip-labels", nargs="*", default=NOT_LINKED, help="entity types not linked")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--keep", choices=["ram", "gpu"], default="ram",
                    help="index in RAM (half precision, search on CPU, a few s) or on the GPU (faster)")
    ap.add_argument("--device", default="cuda", help="encoder and cross-encoder")
    ap.add_argument("--encoder", default=os.environ.get("ENCODER", "bioelx-nlp/bioelx_retriever"))
    ap.add_argument("--cross-encoder", default=os.environ.get("CROSS_ENCODER", "phlobo/xmen-fr-ce-medmentions"))
    for k in ["index", "kb"]:
        v = os.environ.get(k.upper())
        ap.add_argument(f"--{k}", type=Path, default=Path(v) if v else None, required=not v, help=f"default: ${k.upper()}")
    a = ap.parse_args()

    text = a.report.read_text(encoding="utf-8")
    entities = json.loads(a.entities.read_text(encoding="utf-8"))
    todo = [e for e in entities if e["label"] not in a.skip_labels]
    mentions = list(dict.fromkeys(e["text"] for e in todo))        # distinct texts, first-occurrence order
    print(f"{len(entities)} entities, {len(todo)} to link, {len(mentions)} distinct mentions", flush=True)
    if not mentions:
        raise SystemExit("nothing to link")
    if not (a.index / "DONE").exists():
        raise SystemExit(f"index not ready (no DONE file): {a.index}")

    t0 = time.time()
    retriever = Retriever(Encoder(a.encoder, a.device), a.index, keep=a.keep)
    print(f"retriever loaded ({time.time() - t0:.0f} s)", flush=True)
    t0 = time.time()
    found = dict(zip(mentions, retriever(mentions)))
    print(f"retrieval: {time.time() - t0:.1f} s", flush=True)
    concepts = read_concepts(a.kb, {c for r in found.values() for c, _ in r})

    reranked = {}                    # per entity (the context differs between occurrences of the same text)
    if not a.no_rerank:
        ce = CrossEncoder(a.cross_encoder, a.device)
        pairs, where = [], []
        for i, e in enumerate(todo):
            ctx = marked(text, e["start"], e["end"])
            for c, _ in found[e["text"]][:a.rerank_k]:
                if c in concepts:
                    pairs.append((ctx, candidate_text(concepts[c])))
                    where.append((i, c))
        t0 = time.time()
        scores = ce(pairs)
        print(f"cross-encoder: {len(pairs):,} pairs, {time.time() - t0:.1f} s", flush=True)
        for (i, c), s in zip(where, scores):
            reranked.setdefault(i, {})[c] = s

    def top(pairs):
        return [{"cui": c, "name": concepts.get(c, {}).get("canonical_name", ""), "score": round(float(s), 4)}
                for c, s in pairs[:a.top]]

    records, i = [], 0
    for e in entities:
        rec = dict(e)
        if e["label"] not in a.skip_labels:
            ret = found[e["text"]]
            rec["retriever"] = top(ret)
            final = rec["retriever"][0]
            if not a.no_rerank:
                sc = reranked.get(i, {})
                head = sorted(sc.items(), key=lambda x: -x[1])
                rest = [(c, s) for c, s in ret if c not in sc]
                rec["cross_encoder"] = top(head + rest)
                final = rec["cross_encoder"][0]
                rec["retriever_rank_of_final"] = 1 + [c for c, _ in ret].index(final["cui"])
            rec.update(cui=final["cui"], name=final["name"], decided_by="retriever" if a.no_rerank else "cross_encoder")
            i += 1
        records.append(rec)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{a.out}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["start", "end", "text", "label", "ner_score", "cui", "name", "retriever_cui", "retriever_name",
            "retriever_rank_of_final"]
    with open(f"{a.out}.tsv", "w", newline="", encoding="utf-8") as h:
        w = csv.writer(h, delimiter="\t")
        w.writerow(cols)
        for r in records:
            ret = r.get("retriever", [{}])[0]
            w.writerow([r["start"], r["end"], r["text"], r["label"], r["score"], r.get("cui", ""), r.get("name", ""),
                        ret.get("cui", ""), ret.get("name", ""), r.get("retriever_rank_of_final", "")])
    width = max(len(r["text"]) for r in records)
    print()
    for r in records:
        if "cui" in r:
            moved = (f"   (retriever: {r['retriever'][0]['cui']} {r['retriever'][0]['name']})"
                     if r.get("retriever_rank_of_final", 1) != 1 else "")
            print(f"{r['text']:<{width}}  {r['label']:<18} {r['cui']}  {r['name']}{moved}")
        else:
            print(f"{r['text']:<{width}}  {r['label']:<18} (not linked)")
    print(f"\nwritten: {a.out}.json, {a.out}.tsv")


if __name__ == "__main__":
    main()
