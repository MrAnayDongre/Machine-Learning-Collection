from langchain.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain_core.runnables.config import run_in_executor

from sqlalchemy.orm import Session


class VectorStore(PGVector):
    def _get_all_ids(self) -> list[str]:
        with Session(self._bind) as session:
            results = session.query(self.EmbeddingStore.custom_id).all()
            return [result[0] for result in results if result[0] is not None]

    def _get_documents_by_ids(self, ids: list[str]) -> list[Document]:

        with Session(self._bind) as session:
            results = (
                session.query(self.EmbeddingStore)
                .filter(self.EmbeddingStore.custom_id.in_(ids))
                .all()
            )
            return [
                Document(page_content=result.document, metadata=result.cmetadata or {})
                for result in results
                if result.custom_id in ids
            ]

    async def get_all_ids(self) -> list[str]:
        return await run_in_executor(None, self._get_all_ids)

    async def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        return await run_in_executor(None, self._get_documents_by_ids, ids)

    async def delete(
        self, ids: list[str], collection_only: bool = False, **kwargs: any
    ) -> None:
        await run_in_executor(None, super().delete, ids, collection_only, **kwargs)
