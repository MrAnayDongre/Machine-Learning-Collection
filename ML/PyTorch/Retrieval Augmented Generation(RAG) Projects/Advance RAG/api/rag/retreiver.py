from typing import Annotated
from fastapi import Depends
from api.rag.vectordb import VectorDbFactory
from langchain_core.vectorstores import VectorStoreRetriever


def _retriever_factory(
    vector_db: VectorDbFactory,
) -> VectorStoreRetriever:
    return vector_db.as_retriever()


RetrieverFactory = Annotated[VectorStoreRetriever, Depends(_retriever_factory)]
