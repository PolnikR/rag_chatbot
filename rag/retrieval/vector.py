from openai import OpenAI

from rag.models import Chunk, SearchResult


class OpenAIEmbeddingClient:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model

    def embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding


class ChromaVectorRetriever:
    def __init__(self, collection, embedding_client: OpenAIEmbeddingClient):
        self.collection = collection
        self.embedding_client = embedding_client

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        query_embedding = self.embedding_client.embed(query)

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if filters:
            kwargs["where"] = filters

        raw = self.collection.query(**kwargs)

        ids = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        results = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            results.append(SearchResult(
                chunk_id=chunk_id,
                text=text,
                score=1.0 - distance,
                metadata={
                    **metadata,
                    "distance": distance,
                },
            ))

        return results


def load_chunks_from_chroma(collection) -> list[Chunk]:
    raw = collection.get(include=["documents", "metadatas"])

    chunks = []
    for chunk_id, text, metadata in zip(raw["ids"], raw["documents"], raw["metadatas"]):
        chunks.append(Chunk(
            id=chunk_id,
            text=text,
            metadata=metadata or {},
        ))

    return chunks