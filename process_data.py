import json
import uuid

from qdrant_client.http.models import ScoredPoint

from chunking import ChunkingProcessor, FixedSizeChunking, FullSentenceChunking
from constants import METRIC_DOT, METRIC_COSINE, METRIC_EUCLID, METRIC_MANHATTAN, BASE_COLLECTION_NAME, CONTAINER_PORT, \
    CONTAINER_URL, MODEL_NAME
from qdrant_manager import QdrantManager
from questions import ai_questions
from readers import DocumentReader, get_document_reader, Document
from special_unicode_char_removal import remove_special_unicode_characters

ALL_METRICS = [METRIC_DOT, METRIC_COSINE, METRIC_EUCLID, METRIC_MANHATTAN]


def article_parser(articles: Document):
    articles_content: json = json.loads(articles.content)
    documents = []
    child_documents = []

    for article in articles_content:

        parent_id = uuid.uuid5(uuid.NAMESPACE_OID, article.get('file_name'))

        pages = article.get('pages')
        pages_content = []

        for page in pages:
            page_content = page.get('text')

            clean_page_content = remove_special_unicode_characters(page_content)

            pages_content.append(clean_page_content)

            child_document = Document(
                document_id=uuid.uuid5(uuid.NAMESPACE_OID, clean_page_content[:256]),
                content=clean_page_content,
                source=article.get('file_name'),
                doc_type='article_page',
                parent_id=parent_id
            )

            child_documents.append(child_document)

        document = Document(
            document_id=parent_id,
            content=' '.join(pages_content),
            source=article.get('file_name'),
            doc_type='article',
        )

        documents.append(document)

    print('Number of resulting articles: ', len(documents))
    return documents, child_documents


def process_articles(qdrant_manager_instance: QdrantManager):
    processed_articles, processed_pages = read_articles()
    print(processed_articles[0])
    print(processed_pages[0])

    add_base_articles_to_qdrant(qdrant_manager_instance=qdrant_manager_instance, documents=processed_articles)
    add_page_chunked_articles_to_qdrant(qdrant_manager_instance=qdrant_manager_instance, documents=processed_pages)
    add_sentence_chunked_articles_to_qdrant(qdrant_manager_instance=qdrant_manager_instance, documents=processed_pages)


def read_articles():
    reader: DocumentReader = get_document_reader('json')

    document: Document = reader.read('./articles.json')

    documents, child_documents = article_parser(document)

    print('Number of articles: ', len(documents))
    print('Number of pages: ', len(child_documents))

    return documents, child_documents


def create_collection_and_add_documents(qdrant_manager_instance: QdrantManager, collection_name: str,
                                        documents: [Document], metric: str):
    print(f"Adding documents to {collection_name}")
    qdrant_manager_instance.delete_collection(collection_name)
    qdrant_manager_instance.create_collection(collection_name, distance_metric=metric)

    for document in documents:
        qdrant_manager_instance.add_document(document, collection_name)

    print(f"Documents to {collection_name} were added")


def apply_sentence_chunking_to_documents(documents: [Document]):
    chunking_method = FullSentenceChunking()

    chunking_processor = ChunkingProcessor(chunking_method)

    result = chunking_processor.process(documents)

    for point in result:
        print(point)

    return result


def add_base_articles_to_qdrant(qdrant_manager_instance: QdrantManager, documents: [Document]):
    for metric in ALL_METRICS:
        article_collection_name = f'{BASE_COLLECTION_NAME}s_{metric.lower()}'
        create_collection_and_add_documents(qdrant_manager_instance, article_collection_name,
                                            documents=documents, metric=metric)


def add_page_chunked_articles_to_qdrant(qdrant_manager_instance: QdrantManager, documents: [Document]):
    for metric in ALL_METRICS:
        article_pages_collection_name = f'{BASE_COLLECTION_NAME}_pages_{metric.lower()}'
        create_collection_and_add_documents(qdrant_manager_instance, article_pages_collection_name,
                                            documents=documents, metric=metric)

def add_sentence_chunked_articles_to_qdrant(qdrant_manager_instance: QdrantManager, documents: [Document]):
    documents_sentence_chunked = apply_sentence_chunking_to_documents(documents)

    articles_chunked_collection_name = 'articles_sentence_chunking'

    create_collection_and_add_documents(qdrant_manager_instance=qdrant_manager_instance,
                                        collection_name=articles_chunked_collection_name,
                                        documents=documents_sentence_chunked,
                                        metric=METRIC_COSINE)

def distance_experiment(qdrant_manager_instance: QdrantManager):
    collection_names = [collection.name for collection in qdrant_manager_instance.get_client_collections().collections]

    distance_metric_results = {}

    for collection_name in collection_names:
        scores = []
        for question in ai_questions:
            scoredPoint: ScoredPoint = qdrant_manager.search(collection_name=collection_name, query=question, top_k=1)[
                0]
            scores.append(scoredPoint.score)

        distance_metric_results[collection_name] = sum(scores) / len(scores)

    print(distance_metric_results)


if __name__ == '__main__':
    qdrant_manager = QdrantManager(CONTAINER_URL, CONTAINER_PORT, MODEL_NAME)

    process_articles(qdrant_manager)

    distance_experiment(qdrant_manager)
