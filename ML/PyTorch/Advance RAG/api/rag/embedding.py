from typing import Annotated
from fastapi import Depends
from api.settings import settings
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings


def _embedding_factory() -> Embeddings:
    return OllamaEmbeddings(
        base_url=settings.OLLAMA_SERVER, model=settings.EMBEDDING_MODEL
    )


EmbeddingFactory = Annotated[Embeddings, Depends(_embedding_factory)]
