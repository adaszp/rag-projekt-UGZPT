from abc import ABC, abstractmethod
from typing import List

from readers import Document


class ChunkingMethod(ABC):
    """
    Abstract base class for all chunking methods.
    """

    @abstractmethod
    def chunk(self, document: Document) -> List[Document]:
        """
        Abstract method to chunk a single document into smaller documents.

        Args:
            document (Document): The document to chunk.

        Returns:
            List[Document]: A list of chunked documents.
        """
        pass
