from typing import List
from src.document_loader import Document


class TextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            chunks.extend(self._split_text(doc.content, doc.metadata))
        return chunks

    def _split_text(self, text: str, metadata: dict) -> List[Document]:
        chunks = []
        start = 0
        chunk_idx = 0
        step = max(1, self.chunk_size - self.chunk_overlap)
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append(Document(
                    content=chunk_text,
                    metadata={**metadata, "chunk_idx": chunk_idx},
                ))
                chunk_idx += 1
            start += step
        return chunks
