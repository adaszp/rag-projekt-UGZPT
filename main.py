import json

from sentence_transformers import SentenceTransformer

from constants import CONTAINER_URL, CONTAINER_PORT, BASE_COLLECTION_NAME, MODEL_NAME_PARAPHRASE_MINILM
from generation import ResponseGenerator
from qdrant_manager import QdrantManager
from qdrant_retrieval import QdrantRetriever


def qdrant_search_loop(qdrant_manager_instance: QdrantManager, collection_name: str = 'article_pages_cosine'):
    while True:
        user_query = str(input("Enter query you want to search in collection (enter q to quit): "))
        if user_query == 'q':
            break

        results = qdrant_manager_instance.search(collection_name, user_query)
        for idx, result in enumerate(results):
            print(f"{idx + 1}. Document id:{result.id}\nScore: {result.score}\nContent: {result.payload['content']}")


def question_loop(retriever: QdrantRetriever, generator: ResponseGenerator):
    while True:
        user_query = str(input("Enter query you want to search in collection (enter q to quit): "))
        if user_query == 'q':
            break
        retrieved_docs = retriever.get_relevant_documents(user_query)
        answer = generator.generate_response(retrieved_docs, user_query)
        
        try :
            
            strValidator = generator.validate_response(retrieved_docs, user_query, answer.content)

            answer = generator.generate_response(retrieved_docs, f"""
                                                 this is feedback from validation assistant: {strValidator.content}, 
                                                 please consider the suggestion from him and return final answer on the question""")
        
        except :
            print('error')
                
        print("\nANSWER")
        print(answer.content)


if __name__ == "__main__":
    embedding_model = SentenceTransformer(MODEL_NAME_PARAPHRASE_MINILM, cache_folder='./model_cache')
    qdrant_manager = QdrantManager(CONTAINER_URL, CONTAINER_PORT, embedding_model)
    generator = ResponseGenerator('llama3.2:3b-instruct-fp16')
    retriever = QdrantRetriever(qdrant_manager, 'article_pages_cosine')
    
    question_loop(retriever, generator)
