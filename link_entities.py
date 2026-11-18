"""Step 2: link the entities found by Report-Extractor to UMLS concepts.

    python link_entities.py ENTITIES.json --out outputs/report   [--no-rerank] [--top 5]

ENTITIES.json is the output of `extract_medical_entities.py --json` (text, label, score, start, end). Each distinct
mention text is linked once: the retriever's top 100 concepts in a catalogue index, then the reranker's order of those
100. The final concept is the reranker's first, or the retriever's first with --no-rerank; both orders are written.

Writes OUT.json (one record per entity, with the top --top of both orders) and OUT.tsv (one line per entity).
Paths and the linking package come from environment variables (see local.env.example); run.sh loads local.env.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
from pathlib import Path

NOT_LINKED = ["Dosage"]          # a quantity, not a concept
PATHS = ["index", "checkpoint", "graph", "mrconso"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entities", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="prefix: writes OUT.json and OUT.tsv")
    ap.add_argument("--no-rerank", action="store_true", help="final concept = the retriever's first")
    ap.add_argument("--skip-labels", nargs="*", default=NOT_LINKED, help="entity types not linked")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--keep", choices=["ram", "gpu"], default="ram",
                    help="index in RAM (half precision, search on CPU, a few s) or on the GPU (faster)")
    ap.add_argument("--device", default="cuda", help="encoder and reranker (small)")
    ap.add_argument("--encoder", default=os.environ.get("ENCODER", "bioelx-nlp/bioelx_retriever"))
    for k in PATHS:
        v = os.environ.get(k.upper())
        ap.add_argument(f"--{k}", type=Path, default=Path(v) if v else None, required=not v,
                        help=f"default: ${k.upper()}")
    a = ap.parse_args()

    if not os.environ.get("LINKER_MODULE"):
        raise SystemExit("set LINKER_MODULE (the Python module providing Linker), see local.env.example")
    Linker = importlib.import_module(os.environ["LINKER_MODULE"]).Linker

    entities = json.loads(a.entities.read_text(encoding="utf-8"))
    todo = [e for e in entities if e["label"] not in a.skip_labels]
    mentions = list(dict.fromkeys(e["text"] for e in todo))        # distinct texts, first-occurrence order
    print(f"{len(entities)} entities, {len(todo)} to link, {len(mentions)} distinct mentions", flush=True)
    if not mentions:
        raise SystemExit("nothing to link")
    if not (a.index / "DONE").exists():
        raise SystemExit(f"index not ready (no DONE file): {a.index}")

    linker = Linker(a.encoder, a.index, a.checkpoint, a.graph, a.mrconso, device=a.device, keep=a.keep,
                    threads=int(os.environ.get("OMP_NUM_THREADS", 8)))
    results, t = linker(mentions)
    print(f"linking: encode {t['encode']:.2f} s, search {t['search']:.2f} s, rerank {t['rerank']:.2f} s", flush=True)
    by_text = {r["mention"]: r for r in results}
    names = linker.names

    def top(pairs):
        return [{"cui": c, "name": names.get(c, ""), "score": round(float(s), 4)} for c, s in pairs[:a.top]]

    records = []
    for e in entities:
        rec = dict(e)
        r = by_text.get(e["text"])
        if r is not None:
            ret, rer = top(r["retriever"]), top(r["module"])
            final = ret[0] if a.no_rerank else rer[0]
            moved = r["retriever_rank_of_module_top1"]
            rec.update(cui=final["cui"], name=final["name"], decided_by="retriever" if a.no_rerank else "reranker",
                       reranker_changed_top1=moved != 1, retriever_rank_of_reranker_top1=moved,
                       retriever=ret, reranker=rer)
        records.append(rec)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{a.out}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    cols = ["start", "end", "text", "label", "ner_score", "cui", "name", "retriever_cui", "retriever_name",
            "reranker_changed_top1"]
    with open(f"{a.out}.tsv", "w", newline="", encoding="utf-8") as h:
        w = csv.writer(h, delimiter="\t")
        w.writerow(cols)
        for r in records:
            linked = "cui" in r
            w.writerow([r["start"], r["end"], r["text"], r["label"], r["score"],
                        r.get("cui", ""), r.get("name", ""),
                        r["retriever"][0]["cui"] if linked else "", r["retriever"][0]["name"] if linked else "",
                        r.get("reranker_changed_top1", "")])
    width = max(len(r["text"]) for r in records)
    print()
    for r in records:
        if "cui" in r:
            moved = (f"   (retriever: {r['retriever'][0]['cui']} {r['retriever'][0]['name']})"
                     if r["reranker_changed_top1"] and not a.no_rerank else "")
            print(f"{r['text']:<{width}}  {r['label']:<18} {r['cui']}  {r['name']}{moved}")
        else:
            print(f"{r['text']:<{width}}  {r['label']:<18} (not linked)")
    print(f"\nwritten: {a.out}.json, {a.out}.tsv")


if __name__ == "__main__":
    main()
