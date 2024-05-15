from typing import Annotated
from fastapi import Depends

from api.settings import settings
from api.rag.embedding import EmbeddingFactory
from api.tools.vector_store import VectorStore


def _vector_db_factory(
    embedding: EmbeddingFactory,
) -> VectorStore:
    return VectorStore(
        embedding_function=embedding,
        connection_string=settings.database_uri(),
    )


VectorDbFactory = Annotated[VectorStore, Depends(_vector_db_factory)]
