from typing import List

from chunking.chunking_method import ChunkingMethod
from readers import Document


class ChunkingProcessor:
    """
    Processes a list of documents using a specified chunking method.
    """

    def __init__(self, chunking_method: ChunkingMethod):
        """
        Initializes the processor with a chunking method.

        Args:
            chunking_method (ChunkingMethod): The chunking method to use.
        """
        self.chunking_method = chunking_method

    def process(self, documents: List[Document]) -> List[Document]:
        """
        Applies the chunking method to a list of documents.

        Args:
            documents (List[Document]): List of documents to chunk.

        Returns:
            List[Document]: List of chunked documents.
        """
        chunked_documents = []
        for document in documents:
            chunked_documents.extend(self.chunking_method.chunk(document))
        return chunked_documents
