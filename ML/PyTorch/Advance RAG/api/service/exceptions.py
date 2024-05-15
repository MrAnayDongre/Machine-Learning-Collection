from fastapi import HTTPException


class DocumentTypeNotSupported(HTTPException):
    def __init__(self, document_type: str):
        super().__init__(
            status_code=400, detail=f"Document type {document_type} is not supported."
        )
