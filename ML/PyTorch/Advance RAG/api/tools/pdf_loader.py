from typing import AsyncIterator, Iterator
from langchain.document_loaders.pdf import PyMuPDFLoader
from langchain_core.documents import Document
from langchain.document_loaders.parsers.pdf import PyMuPDFParser
from langchain.document_loaders.blob_loaders import Blob
from langchain_core.runnables import run_in_executor


class PDFLoader(PyMuPDFLoader):
    def __init__(
        self,
        file_bytes: bytes,
        *,
        extract_images: bool = False,
        **kwargs: any,
    ) -> None:
        """Load `PDF` files using `PyMuPDF` from a BytesIO stream."""
        try:
            import fitz  # noqa:F401
        except ImportError:
            raise ImportError(
                "`PyMuPDF` package not found, please install it with "
                "`pip install pymupdf`"
            )
        self.file_bytes = file_bytes
        self.extract_images = extract_images
        self.text_kwargs = kwargs

    def load(self) -> Iterator[Document]:
        parser = PyMuPDFParser(
            text_kwargs=self.text_kwargs, extract_images=self.extract_images
        )

        blob = Blob.from_data(self.file_bytes, path="bytes")

        yield from parser.lazy_parse(blob)

    async def aload(self):
        return [document async for document in self.alazy_load()]

    async def alazy_load(self) -> AsyncIterator[Document]:
        iterator = await run_in_executor(None, self.load)
        done = object()
        while True:
            doc = await run_in_executor(None, next, iterator, done)  # type: ignore[call-arg, arg-type]
            if doc is done:
                break
            yield doc
