import uuid
from typing import List

from chunking.chunking_method import ChunkingMethod
from readers import Document


class FixedSizeChunking(ChunkingMethod):
    """
    A simple chunking method that splits the document into chunks of a fixed size.
    """

    def __init__(self, chunk_size: int = 200):
        """
        Initializes the chunking method with a fixed chunk size.

        Args:
            chunk_size (int): The maximum number of characters per chunk.
        """
        self.chunk_size = chunk_size

    def chunk(self, document: Document) -> List[Document]:
        """
        Splits the document into chunks of fixed size.

        Args:
            document (Document): The document to chunk.

        Returns:
            List[Document]: A list of chunked documents.
        """
        chunks = []
        content = document.content
        for i in range(0, len(content), self.chunk_size):
            chunk_content = content[i:i+self.chunk_size]
            chunk_name = f"{document.document_id}_chunk_{i // self.chunk_size}"
            chunk = Document(
                document_id=uuid.uuid5(namespace=uuid.NAMESPACE_OID, name=chunk_name),
                content=chunk_content,
                source=document.metadata["source"],
                doc_type=document.metadata["type"],
                last_modified=document.metadata["last_modified"],
                parent_id=document.document_id
            )
            chunks.append(chunk)
        return chunks
