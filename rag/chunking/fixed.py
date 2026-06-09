from rag.models import Document, Chunk


class FixedTokenChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents: list[Document]) -> list[Chunk]:
        chunks = []

        for doc in documents:
            words = doc.text.split()
            step = self.chunk_size - self.overlap

            for start in range(0, len(words), step):
                part = words[start:start + self.chunk_size]
                if not part:
                    continue

                chunk_id = f"{doc.id}:chunk:{len(chunks)}"
                chunks.append(Chunk(
                    id=chunk_id,
                    text=" ".join(part),
                    metadata={
                        **doc.metadata,
                        "document_id": doc.id,
                        "chunk_start": start,
                        "chunk_size": len(part),
                    },
                ))

        return chunks