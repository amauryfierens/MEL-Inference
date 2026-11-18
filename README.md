# MEL-Inference

French medical report → medical entities → UMLS concepts.

1. **Extraction**: [Report-Extractor](https://github.com/TheodoreCouz/Report-Extractor) (git submodule), which runs
   GLiNER `almanach/camembert-bio-gliner-v0.1` on CPU. It finds 8 entity types: Maladie, Symptôme, Médicament,
   Dosage, Examen, Procédure, Anatomie and Résultat d'examen.
2. **Retrieval**: [BioELX](https://huggingface.co/bioelx-nlp/bioelx_retriever) returns the 100 closest UMLS concepts
   for each distinct mention. It does an exact search over every French and English UMLS name.
3. **Reranking**: xMEN's published cross-encoder (`phlobo/xmen-fr-ce-medmentions`) reads each mention in its context
   and reorders the retriever's first 64 candidates.

The final concept is the cross-encoder's first, or the retriever's first with `--no-rerank`. Both orders are written.
Dosage entities are not linked.

## Setup

The UMLS is needed. Its licence is free but must be requested from the NLM, and the UMLS cannot be redistributed:
https://www.nlm.nih.gov/research/umls/

```
git clone --recurse-submodules https://github.com/amauryfierens/MEL-Inference.git && cd MEL-Inference
pip install -r Report-Extractor/requirements.txt -r requirements.txt
python build_kb.py --umls /path/to/UMLS/META --lang fr --out data/kb_fr.jsonl     # xMEN's builder
python build_index.py --kb data/kb_fr.jsonl --out data/index_bioelx_fr               # GPU, a few hours
cp local.env.example local.env                                                       # fill in the paths
```

## Use

```
bash run.sh REPORT.txt [GPU] [--no-rerank] [--top 10] [--rerank-k 100] [--keep gpu]
```

The outputs go to `outputs/`:
- `<name>_entities.json`: the extracted entities;
- `<name>_linked.json`: everything, including the top-k of both orders;
- `<name>_linked.tsv`: one line per entity.

Loading takes about a minute. The index is held in half precision in RAM, or on the GPU with `--keep gpu`. Each
report then takes a few seconds.

## Known limits

- There is no negation handling. In "ni douleur oculaire", *douleur oculaire* is extracted and linked like any other
  mention.
- The cross-encoders were trained by xMEN's authors on MedMentions translated into each language.

## Credits

GLiNER CamemBERT-bio (almanach), BioELX, and xMEN (Borchert et al.). Their models keep their own licences.
