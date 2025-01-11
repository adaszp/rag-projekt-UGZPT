from readers.json_reader import JSONReader
from readers.base_reader import DocumentReader

def get_document_reader(file_type: str) -> DocumentReader:
    readers = {
        'json': JSONReader
    }
    return readers.get(file_type.lower(), JSONReader)()  # Default to TXTReader
