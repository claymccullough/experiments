from typing import List

from semantic_router.encoders import BaseEncoder

from ingest.ingest import get_ollama_embeddings


class OllamaEncoder(BaseEncoder):
    name = "OllamaEncoder"
    score_threshold = 0.5
    type = "ollama"
    embeddings = get_ollama_embeddings()

    def __call__(self, docs: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(docs)
