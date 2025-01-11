import uuid
from typing import Optional
from datetime import datetime

class Document:
    """
    A class to represent a document with metadata and content for RAG systems.
    """

    def __init__(
        self,
        document_id: int | uuid.UUID,
        content: str,
        source: str,
        doc_type: str,
        last_modified: Optional[str] = None,
        parent_id: Optional[int | uuid.UUID] = None
    ):
        """
        Initializes the Document object.

        Args:
            document_id (str): Unique identifier for the document.
            content (str): Full content of the document as plain text.
            source (str): File name or source URL.
            doc_type (str): Document type (e.g., 'json', 'txt', 'pdf').
            last_modified (Optional[str]): Parsing last_modified. Defaults to current UTC time.
            parent_id (Optional[str]): ID of the parent document. Defaults to None.
        """
        self.document_id = document_id
        self.content = content
        self.parent_id = parent_id
        self.source = source
        self.type = doc_type
        self.last_modified = last_modified or datetime.now().isoformat()

    def to_dict(self) -> dict:
        """
        Converts the Document instance to a dictionary.

        Returns:
            dict: A dictionary representation of the Document instance.
        """
        return {
            "document_id": self.document_id,
            "parent_id": self.parent_id,
            "source": self.source,
            "type": self.type,
            "last_modified": self.last_modified,
            "content": self.content
        }

    def __repr__(self) -> str:
        """
        Returns a string representation of the Document instance.
        """
        summary_len = 250
        if len(self.content) > summary_len:
            content_summary = self.content[:summary_len] + '...'
        else:
            content_summary = self.content
        return f"Document( document_id={self.document_id}, parent_id={self.parent_id}, source={self.source}, type={self.type}, last_modified={self.last_modified}, content={content_summary}"
