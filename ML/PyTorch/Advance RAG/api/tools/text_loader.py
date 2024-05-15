import chardet
from typing import AsyncIterator, Iterator
from langchain_core.documents import Document
from langchain.document_loaders.text import TextLoader as LangchainTextLoader
from langchain_community.document_loaders.helpers import FileEncoding
from langchain_core.runnables import run_in_executor


class TextLoader(LangchainTextLoader):
    def __init__(
        self,
        file_bytes: bytes,
        encoding: str = "utf-8",
        autodetect_encoding: bool = True,
    ):
        """Initialize with file path."""
        self.file_bytes = file_bytes
        self.encoding = encoding
        self.autodetect_encoding = autodetect_encoding

    def detect_file_encodings(self) -> list[FileEncoding]:
        return chardet.detect_all(self.file_bytes)

    def load(self) -> Iterator[Document]:
        text = ""

        try:
            text = self.file_bytes.decode(self.encoding)
        except UnicodeDecodeError as e:
            if self.autodetect_encoding:
                detected_encodings = self.detect_file_encodings()
                for encoding in detected_encodings:
                    try:
                        text = self.file_bytes.decode(encoding.encoding)
                        break
                    except UnicodeDecodeError:
                        continue
            else:
                raise RuntimeError("Error decoding text file") from e

        metadata = {"source": "stream"}
        yield Document(page_content=text, metadata=metadata)

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
