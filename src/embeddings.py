from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str = "paraphrase-multilingual-mpnet-base-v2"):
        print(f"   埋め込みモデルを読み込み中: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension: int = self.model.get_sentence_embedding_dimension()
        print(f"   次元数: {self.dimension}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 20,
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True)[0]
