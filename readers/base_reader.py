from abc import ABC, abstractmethod
from readers.document import Document
from typing import Optional

class DocumentReader(ABC):
    @abstractmethod
    def read(self, file_path: str, parent_id: Optional[str] = None) -> Document:
        pass
