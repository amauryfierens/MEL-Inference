# MEL-Inference

French medical report → medical entities → UMLS concepts.

1. **Extraction** — [Report-Extractor](https://github.com/TheodoreCouz/Report-Extractor) (git submodule): GLiNER
   `almanach/camembert-bio-gliner-v0.1`, 8 entity types (Maladie, Symptôme, Médicament, Dosage, Examen, Procédure,
   Anatomie, Résultat d'examen), on CPU.
2. **Linking** — `link_entities.py`: the retriever (BioELX by default) returns the top 100 UMLS concepts of each
   distinct mention in a catalogue index, then a reranker reorders them. The final concept is the reranker's first, or
   the retriever's first with `--no-rerank`; both orders are written. Dosage entities are not linked.

## Use

```
git clone --recurse-submodules https://github.com/amauryfierens/MEL-Inference.git
cp local.env.example local.env      # paths to the Python environments, index, reranker, UMLS
bash run.sh REPORT.txt [GPU] [--no-rerank] [--top 10] [--keep gpu]
```

Outputs in `outputs/`: `<name>_entities.json` (step 1), `<name>_linked.json` (everything, top-k of both orders) and
`<name>_linked.tsv` (one line per entity).

The linking step expects a Python module (`LINKER_MODULE`) exposing a `Linker` class: built once with the encoder,
index, reranker checkpoint, UMLS graph and MRCONSO, then called on a list of mentions.

## Known limits

- No negation handling: in "ni douleur oculaire", *douleur oculaire* is extracted and linked like any other mention.
- Concept names shown are the UMLS preferred English names.
