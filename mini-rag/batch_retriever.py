import faiss
import numpy as np
from typing import Iterable, Tuple, Optional

def l2_normalize_inplace(x: np.ndarray) -> None:
    faiss.normalize_L2(x)

class FaissRetriever:
    def __init__(
        self,
        dim: int,
        cosine: bool = True,
        nlist: int = 4096,      # IVF clusters (tune)
        m: int = 64,            # PQ subquantizers (tune)
        nbits: int = 8,         # bits per code (8 is common)
    ):
        self.dim = dim
        self.cosine = cosine

        quantizer = faiss.IndexFlatIP(dim) if cosine else faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

        # If you want explicit ids:
        self.index = faiss.IndexIDMap2(self.index)

        # Search-time knobs
        self.nprobe = 16  # tune: higher = better recall, slower

    def train(self, training_vectors: np.ndarray) -> None:
        x = training_vectors.astype(np.float32, copy=False)
        if self.cosine:
            l2_normalize_inplace(x)
        self.index.train(x)

    def add_batches(self, batches: Iterable[Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        batches yields (X_batch, ids_batch)
        X_batch: (B, dim) float32
        ids_batch: (B,) int64
        """
        if not self.index.is_trained:
            raise RuntimeError("Index must be trained before adding vectors.")

        for Xb, ib in batches:
            Xb = Xb.astype(np.float32, copy=False)
            ib = ib.astype(np.int64, copy=False)
            if self.cosine:
                l2_normalize_inplace(Xb)
            self.index.add_with_ids(Xb, ib)

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        self.index.nprobe = self.nprobe
        q = q.astype(np.float32, copy=False)
        if self.cosine:
            l2_normalize_inplace(q)
        scores, ids = self.index.search(q, k)
        return scores, ids

    def save(self, path: str) -> None:
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str, cosine: bool = True) -> "FaissRetriever":
        index = faiss.read_index(path)
        dim = index.d
        obj = cls(dim=dim, cosine=cosine)
        obj.index = index
        return obj
    

def iter_embedding_batches(transformer, batch_size: int):
    # Example idea: read chunks from disk, embed in batches, yield arrays
    buf_vecs, buf_ids = [], []
    for chunk in transformer.iter_chunks():  # streaming
        vec = transformer.embed_chunk(chunk)  # (dim,)
        buf_vecs.append(vec)
        buf_ids.append(chunk.id)
        if len(buf_vecs) == batch_size:
            yield np.vstack(buf_vecs), np.array(buf_ids, dtype=np.int64)
            buf_vecs, buf_ids = [], []
    if buf_vecs:
        yield np.vstack(buf_vecs), np.array(buf_ids, dtype=np.int64)