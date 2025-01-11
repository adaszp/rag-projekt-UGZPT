from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, MatchValue, Record, ScoredPoint
from qdrant_client.models import PointStruct, Filter
import qdrant_client.http.models as qmodels
from sentence_transformers import SentenceTransformer

from readers import Document


class QdrantManager:
    def __init__(self, host: str, port: int, model_name: str, cache_folder: str = './model_cache'):
        self.client = QdrantClient(host=host, port=port)
        self.model = SentenceTransformer(model_name, cache_folder=cache_folder)
        self.threshold = 0.5

    def get_client_collections(self):
        return self.client.get_collections()

    def get_collection(self, collection_name):
        if any(c.name == collection_name for c in self.get_client_collections().collections):
            return self.client.get_collection(collection_name)

    def get_all_points_from_collection(self, collection_name: str, limit: int = 10):
        if any(c.name == collection_name for c in self.get_client_collections().collections):
            return self.client.scroll(collection_name, limit=limit)

    def create_collection(self, collection_name: str, vector_size: int = 384, distance_metric: str = "Cosine"):
        client_collections = self.get_client_collections().collections
        if any(c.name == collection_name for c in client_collections):
            print(f'Collection {collection_name} already exists')
        else:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=distance_metric),
            )

    def delete_collection(self, collection_name: str):
        client_collections = self.get_client_collections().collections
        if any(c.name == collection_name for c in client_collections):
            self.client.delete_collection(collection_name)

    def add_document(self, document: Document, collection_name: str):
        """
        Adds a Document to the Qdrant collection.
        """
        if isinstance(document.document_id, int):
            point_id = int(document.document_id)
        else:
            point_id = str(document.document_id)

        vector = self.model.encode(document.content).tolist()
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "document_id": document.document_id,
                "parent_id": document.parent_id,
                "source": document.source,
                "type": document.type,
                "last_modified": document.last_modified,
                "content": document.content
            }
        )
        self.client.upsert(collection_name=collection_name, points=[point])

    def search(self, collection_name: str, query, top_k: int = 5, query_filter: Filter = None):
        query_encoded = self.model.encode(query).tolist()
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_encoded,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter
        )

    def auto_merging_retriever(self, collection_name: str, parent_collection_name: str, query, top_k: int = 10,
                               query_filter: Filter = None):
        query_encoded = self.model.encode(query).tolist()
        children_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_encoded,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter
        )

        children_parsed = [parse_point_to_document(entry) for entry in children_result]

        parents = []
        parents_map = {}
        for child in children_result:
            child_document = parse_point_to_document(child)
            parent_id = child_document.parent_id
            if not parent_id:
                continue
            else:
                filter_condition = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=parent_id)
                        )
                    ]
                )
                parent_point = self.client.scroll(
                    collection_name=f"{parent_collection_name}",
                    scroll_filter=filter_condition,
                    limit=1
                )
                parent_document_id = parent_point[0][0].payload["document_id"]
                if parent_document_id not in parents_map:
                    parents_map[parent_document_id] = {"document": parent_point[0][0], "sum_score": child.score}
                else:
                    previous_score = parents_map.get(parent_document_id)['sum_score']
                    parents_map[parent_document_id] = {"document": parent_point[0][0], "sum_score": child.score + previous_score}
                if parent_point not in parents:
                    parents.append(parent_point)

        sorted_records = sorted(parents_map.items(), key=lambda item: item[1]['sum_score'], reverse=True)

        print("SORTED RECORDS:")
        for record in sorted_records:
            print(f'record:\nscore: {record[1]["sum_score"]}\ndocument: {record[1]["document"]}\n')

        return [record[1]["document"] for record in sorted_records]


def parse_point_to_document(point: ScoredPoint) -> Document:
    return Document(
        document_id=point.id,
        source=point.payload['source'],
        doc_type=point.payload['type'],
        content=point.payload['content'],
        parent_id=point.payload['parent_id']
    )