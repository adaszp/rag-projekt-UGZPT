import json
from pathlib import Path
from datetime import datetime
import os
from readers.document import Document
from readers.base_reader import DocumentReader
from typing import Optional
import uuid

class JSONReader(DocumentReader):
    def read(self, file_path: str, parent_id: Optional[int | uuid.UUID] = None, encoding: str = 'utf-8') -> Document:
        with open(file_path, 'r', encoding=encoding) as f:
            content = json.load(f)
        last_modified = datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        return Document(
            document_id=uuid.uuid5(uuid.NAMESPACE_OID, Path(file_path).stem),
            content=json.dumps(content),
            source=file_path,
            doc_type="json",
            parent_id=parent_id,
            last_modified=last_modified
        )
