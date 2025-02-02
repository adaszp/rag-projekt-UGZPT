from langchain.schema import Document
from qdrant_manager import QdrantManager


class QdrantRetriever:
    def __init__(self, qdrant_manager:QdrantManager, collection_name:str, top_k:int=1):
        self.qdrant_manager = qdrant_manager
        self.collection_name = collection_name
        self.top_k = top_k

    def get_relevant_documents(self, query:str):

        # results = self.qdrant_manager.search(self.collection_name, query,top_k=self.top_k)
        results = self.qdrant_manager.auto_merging_retriever(self.collection_name, 'articles_cosine', query,top_k=self.top_k)

        documents = []

        for result in results:
            document = Document(
                page_content=result.payload['content'],
                metadata={
                    "document_id": result.id,
                    # "score": result.score,
                    "source": result.payload.get("source", "unknown"),
                    "type": result.payload.get("type", "unknown"),
                    "last_modified": result.payload.get("last_modified", "unknown"),
                }
            )
            documents.append(document)

        return documents
    
    def get_relevant_chunks(self, query:str):

        results = self.qdrant_manager.search(self.collection_name, query,top_k=self.top_k)

        documents = []

        for result in results:
            documents.append({
              "content": result.payload['content'],
              "score": result.score,
            })

        return documents