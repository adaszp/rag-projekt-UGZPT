import json
import uuid
import time

from qdrant_client.http.models import ScoredPoint
from sentence_transformers import SentenceTransformer

from chunking import ChunkingProcessor, FullSentenceChunking
from constants import METRIC_DOT, METRIC_COSINE, METRIC_EUCLID, METRIC_MANHATTAN, BASE_COLLECTION_NAME, CONTAINER_PORT, \
    CONTAINER_URL, MODEL_NAME_ALL_MINILM, MODEL_NAME_PARAPHRASE_MINILM, MODEL_NAME_DISTILBERT, MODEL_NAME_MPNET, \
    MODEL_NAME_DISTILROBERTA
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


def process_articles(qdrant_manager_instance: QdrantManager, vector_size: int):
    processed_articles, processed_pages = read_articles()

    add_base_articles_to_qdrant(qdrant_manager_instance=qdrant_manager_instance, documents=processed_articles,
                                base_collection_name=BASE_COLLECTION_NAME, vector_size=vector_size)
    add_page_chunked_articles_to_qdrant(qdrant_manager_instance=qdrant_manager_instance, documents=processed_pages,
                                        base_collection_name=BASE_COLLECTION_NAME, vector_size=vector_size)
    # add_sentence_chunked_articles_to_qdrant(qdrant_manager_instance=qdrant_manager_instance, documents=processed_pages)


def process_articles_for_embedding_experiment(qdrant_manager_instance: QdrantManager, base_collection_name: str,
                                              vector_size: int):
    processed_articles, processed_pages = read_articles()

    add_base_articles_to_qdrant(qdrant_manager_instance=qdrant_manager_instance, documents=processed_articles,
                                base_collection_name=base_collection_name, vector_size=vector_size)
    add_page_chunked_articles_to_qdrant(qdrant_manager_instance=qdrant_manager_instance, documents=processed_pages,
                                        base_collection_name=base_collection_name, vector_size=vector_size)


def read_articles():
    reader: DocumentReader = get_document_reader('json')

    document: Document = reader.read('./articles.json')

    documents, child_documents = article_parser(document)

    print('Number of articles: ', len(documents))
    print('Number of pages: ', len(child_documents))

    return documents, child_documents


def create_collection_and_add_documents(qdrant_manager_instance: QdrantManager, collection_name: str,
                                        documents: [Document], metric: str, vector_size: int):
    print(f"Adding documents to {collection_name}")
    qdrant_manager_instance.delete_collection(collection_name)
    qdrant_manager_instance.create_collection(collection_name, distance_metric=metric, vector_size=vector_size)

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


def add_base_articles_to_qdrant(qdrant_manager_instance: QdrantManager, documents: [Document],
                                base_collection_name: str, vector_size: int):
    metric = METRIC_COSINE

    article_collection_name = f'{base_collection_name}_{metric.lower()}'
    create_collection_and_add_documents(qdrant_manager_instance=qdrant_manager_instance,
                                        collection_name=article_collection_name,
                                        documents=documents,
                                        metric=metric,
                                        vector_size=vector_size)


def add_page_chunked_articles_to_qdrant(qdrant_manager_instance: QdrantManager, documents: [Document],
                                        base_collection_name: str, vector_size: int):
    metric = METRIC_COSINE

    article_pages_collection_name = f'{base_collection_name}_pages_{metric.lower()}'
    create_collection_and_add_documents(qdrant_manager_instance=qdrant_manager_instance,
                                        collection_name=article_pages_collection_name,
                                        documents=documents,
                                        metric=metric,
                                        vector_size=vector_size)


def add_sentence_chunked_articles_to_qdrant(qdrant_manager_instance: QdrantManager, documents: [Document],
                                            vector_size: int):
    documents_sentence_chunked = apply_sentence_chunking_to_documents(documents)

    articles_chunked_collection_name = 'articles_sentence_chunking'

    create_collection_and_add_documents(qdrant_manager_instance=qdrant_manager_instance,
                                        collection_name=articles_chunked_collection_name,
                                        documents=documents_sentence_chunked,
                                        metric=METRIC_COSINE,
                                        vector_size=vector_size)


def distance_experiment(qdrant_manager_instance: QdrantManager):
    collection_names = [collection.name for collection in qdrant_manager_instance.get_client_collections().collections]

    distance_metric_results = {}

    for collection_name in collection_names:
        scores = []
        for question in ai_questions:
            scoredPoint: ScoredPoint = qdrant_manager_instance.search(collection_name=collection_name, query=question, top_k=1)[0]
            scores.append(scoredPoint.score)

        distance_metric_results[collection_name] = sum(scores) / len(scores)

    print(distance_metric_results)


def model_embeddings_experiment(model_names: [str]):
    model_results = {}
    for model_name in model_names:

        only_model_name = model_name.split('/')[-1]

        print(f"Testing embeddings for model: {only_model_name}")


        print(f"Creating embedding model: {only_model_name}")
        embedding_model = SentenceTransformer(model_name, cache_folder='./model_cache')
        embedding_model_vector_size = embedding_model.get_sentence_embedding_dimension()
        print(f"Embedding model {only_model_name} created")

        qdrant_manager = QdrantManager(CONTAINER_URL, CONTAINER_PORT, embedding_model)

        base_collection_name = f"articles_{only_model_name}"

        print(f"Processing articles for embedding model: {only_model_name}")
        process_articles_for_embedding_experiment(qdrant_manager_instance=qdrant_manager, vector_size=embedding_model_vector_size, base_collection_name=base_collection_name)
        print(f"Processing articles for embedding model {only_model_name} finished")

        collection_name_articles = f'{base_collection_name}_{METRIC_COSINE.lower()}'
        collection_name_article_pages = f'{base_collection_name}_pages_{METRIC_COSINE.lower()}'

        print(f"Testing embedding model: {only_model_name}")
        articles_avg_score, articles_avg_time = test_qdrant_model_embeddings(qdrant_manager_instance=qdrant_manager, collection_name=collection_name_articles)
        article_pages_avg_score, article_pages_avg_time = test_qdrant_model_embeddings(qdrant_manager_instance=qdrant_manager, collection_name=collection_name_article_pages)
        print(f"Testing embedding model {only_model_name} finished")

        model_results[collection_name_articles] = {'avg_score': articles_avg_score, 'avg_time': articles_avg_time}
        model_results[collection_name_article_pages] = {'avg_score': article_pages_avg_score, 'avg_time': article_pages_avg_time}
        print(model_results[collection_name_articles])
        print(model_results[collection_name_article_pages])


    print(model_results)


def test_qdrant_model_embeddings(qdrant_manager_instance: QdrantManager, collection_name: str):

    scores = []
    times = []
    for question in ai_questions:
        start_time = time.perf_counter()

        scoredPoint: ScoredPoint = qdrant_manager_instance.search(collection_name=collection_name, query=question, top_k=1)[0]

        end_time = time.perf_counter()

        scores.append(scoredPoint.score)
        times.append(end_time - start_time)


    avg_score = sum(scores) / len(scores)
    avg_time = sum(times) / len(times)

    return avg_score, avg_time

if __name__ == '__main__':
    # model_names = [
    #     MODEL_NAME_ALL_MINILM,
    #     MODEL_NAME_PARAPHRASE_MINILM,
    #     MODEL_NAME_DISTILBERT,
    #     MODEL_NAME_MPNET,
    #     MODEL_NAME_DISTILROBERTA
    # ]
    #
    # model_embeddings_experiment(model_names=model_names)

    # Best model based on experiment
    embedding_model = SentenceTransformer(MODEL_NAME_PARAPHRASE_MINILM, cache_folder='./model_cache')
    embedding_model_vector_size = embedding_model.get_sentence_embedding_dimension()

    qdrant_manager = QdrantManager(CONTAINER_URL, CONTAINER_PORT, embedding_model)

    process_articles(qdrant_manager_instance=qdrant_manager, vector_size=embedding_model_vector_size)
    
    distance_experiment(qdrant_manager)

