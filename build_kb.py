"""Build the concept base of one language from the UMLS, in xMEN's format (one JSON line per concept:
concept_id, canonical_name, aliases, types), with xMEN's own builder: names in that language and in English, every
source, every semantic group. Both the retriever's catalogue and the cross-encoder's candidate texts read it.

    pip install xmen
    python build_kb.py --umls /path/to/UMLS/2025AB/META --lang fr --out data/kb_fr.jsonl

The UMLS needs a (free) licence from the NLM: https://www.nlm.nih.gov/research/umls/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LANGUAGES = {"en": ["en"], "fr": ["fr", "en"], "de": ["de", "en"], "es": ["es", "en"], "nl": ["nl", "en"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--umls", type=Path, required=True, help="the META folder (MRCONSO.RRF, MRSTY.RRF, ...)")
    ap.add_argument("--lang", choices=list(LANGUAGES), default="fr")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from xmen.umls import get_umls_concepts
    concepts = get_umls_concepts(str(a.umls), LANGUAGES[a.lang], sabs=[], sources=[], semantic_groups=None,
                                 semantic_types=None, semantic_group_file_version="v03")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for v in concepts.values():
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    tmp.rename(a.out)
    print(f"{len(concepts):,} concepts -> {a.out}")


if __name__ == "__main__":
    main()
