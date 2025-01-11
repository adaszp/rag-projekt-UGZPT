import uuid
from typing import List
import nltk
nltk.download('punkt')

import spacy

from chunking.chunking_method import ChunkingMethod
from readers import Document


class FullSentenceChunking(ChunkingMethod):
    """
    A simple chunking method that splits the document into chunks of a fixed size.
    """

    def __init__(self):
        """
        Initializes the chunking method with a full sentence chunk size.
        """

    def chunk(self, document: Document) -> List[Document]:
        """
        Splits the document into chunks of full sentence size.

        Args:
            document (Document): The document to chunk.

        Returns:
            List[Document]: A list of chunked documents.
        """
        nlp = spacy.load("en_core_web_sm")
        content = document.content
        data_parsed = nlp(content)
        chunks = [sentence.text.strip() for sentence in data_parsed.sents]
        result = []
        for count, chunk in enumerate(chunks):
            if chunk == '':
                continue
            chunk_name = f"{chunk}_{count}"
            chunk_document = Document(
                document_id=uuid.uuid5(namespace=uuid.NAMESPACE_OID, name=chunk_name),
                content=chunk,
                source=document.source,
                doc_type=document.type,
                last_modified=document.last_modified,
                parent_id=document.document_id
            )
            result.append(chunk_document)
        return result