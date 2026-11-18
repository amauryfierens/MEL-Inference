"""Dense retriever: a bi-encoder (BioELX by default) and an exact search over every name of a concept base.

Encoding as published for BioELX: CLS vector, at most 64 tokens, L2-normalised (bf16 on GPU). The index is a folder
with vectors.npy (float16, one row per name) and cuis.json (the concept of each row); an exact search keeps the best
name of each concept and returns the top k concepts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

CHUNK = 400_000


class Encoder:
    def __init__(self, name: str, device: str = "cuda", max_length: int = 64):
        from transformers import AutoModel, AutoTokenizer
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(name, dtype=dtype).to(self.device).eval()
        self.max_length = max_length
        self.dimension = self.model.config.hidden_size

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = np.empty((len(texts), self.dimension), dtype=np.float32)
        for s in range(0, len(texts), batch_size):
            tok = self.tokenizer(texts[s:s + batch_size], padding=True, truncation=True, max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
            v = self.model(**tok).last_hidden_state[:, 0].float()
            out[s:s + batch_size] = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
        return out


def read_kb(path: Path):
    """Yield the concepts of a base in xMEN's format (see build_kb.py)."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_vectors(folder: Path, dimension: int):
    """vectors.npy (this repository's format), or index.faiss (a flat faiss index: the vectors end the file)."""
    folder = Path(folder)
    cuis = json.loads((folder / "cuis.json").read_text())
    if (folder / "vectors.npy").exists():
        return np.load(folder / "vectors.npy", mmap_mode="r"), cuis
    f = folder / "index.faiss"
    n = len(cuis)
    return np.memmap(f, dtype="float32", mode="r", offset=f.stat().st_size - n * dimension * 4, shape=(n, dimension)), cuis


class Retriever:
    def __init__(self, encoder: Encoder, index: Path, keep: str = "ram", k: int = 100, keep_names: int = 2048):
        from tqdm import tqdm
        self.encoder, self.k = encoder, k
        vec, self.cuis = load_vectors(index, encoder.dimension)
        self.store = torch.device("cuda") if keep == "gpu" else torch.device("cpu")
        self.vectors = torch.empty(vec.shape, dtype=torch.float16, device=self.store)
        for s in tqdm(range(0, len(vec), CHUNK), desc="index -> memory", unit="chunk"):
            self.vectors[s:s + CHUNK] = torch.from_numpy(np.asarray(vec[s:s + CHUNK], dtype=np.float16)).to(self.store)
        self.keep_names = min(keep_names, len(self.cuis))

    def __call__(self, mentions: list[str]):
        """Top k concepts of each mention (best name per concept) and their cosine scores."""
        q = torch.from_numpy(self.encoder.encode(mentions, 64))
        if self.store.type == "cuda":
            top_s, top_i = (q.to(self.store, torch.float16) @ self.vectors.T).float().topk(self.keep_names, dim=1)
        else:
            top_s = torch.full((len(q), self.keep_names), -float("inf"))
            top_i = torch.zeros((len(q), self.keep_names), dtype=torch.long)
            for s in range(0, len(self.vectors), CHUNK):
                sc = q @ self.vectors[s:s + CHUNK].float().T
                cs, ci = sc.topk(min(self.keep_names, sc.shape[1]), dim=1)
                top_s, pos = torch.cat([top_s, cs], 1).topk(self.keep_names, dim=1)
                top_i = torch.cat([top_i, ci + s], 1).gather(1, pos)
        top_s, top_i = top_s.cpu().numpy(), top_i.cpu().numpy()
        out = []
        for r in range(len(q)):
            seen, res = set(), []
            for j in range(top_s.shape[1]):
                c = self.cuis[top_i[r, j]]
                if c not in seen:
                    seen.add(c)
                    res.append((c, float(top_s[r, j])))
                    if len(res) == self.k:
                        break
            out.append(res)
        return out
