import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from src.document_loader import Document


class VectorStore:
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        # IndexFlatIP + normalized vectors → cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document], embeddings: np.ndarray) -> None:
        self.documents.extend(documents)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Document, float]]:
        query = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(k, self.index.ntotal)
        if k == 0:
            return []
        scores, indices = self.index.search(query, k)
        return [
            (self.documents[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str) -> None:
        self.index = faiss.read_index(path + ".faiss")
        with open(path + ".pkl", "rb") as f:
            self.documents = pickle.load(f)
