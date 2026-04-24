from typing import List

from src.document_loader import Document
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class Retriever:
    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        query_embedding = self.embedding_model.embed_query(query)
        results = self.vector_store.search(query_embedding, k=k)
        return [doc for doc, _ in results]

    def retrieve_for_sections(
        self, queries: List[str], k_per_query: int = 3
    ) -> str:
        """Retrieve unique chunks for multiple section-level queries and format as context."""
        seen: dict = {}
        for query in queries:
            for doc in self.retrieve(query, k=k_per_query):
                key = (
                    doc.metadata.get("source", ""),
                    doc.metadata.get("chunk_idx", 0),
                )
                if key not in seen:
                    seen[key] = doc

        parts = []
        for doc in seen.values():
            filename = doc.metadata.get("filename", "unknown")
            parts.append(f"【出典: {filename}】\n{doc.content}")

        return "\n\n---\n\n".join(parts)
