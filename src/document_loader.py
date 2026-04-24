from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class Document:
    content: str
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    def load_directory(self, directory: str) -> List[Document]:
        docs = []
        path = Path(directory)
        if not path.exists():
            return docs
        for file_path in sorted(path.glob("*.md")):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:
                docs.append(Document(
                    content=content,
                    metadata={"source": str(file_path), "filename": file_path.name},
                ))
        return docs
