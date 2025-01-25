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



if __name__ == "__main__":
    embedding_model = SentenceTransformer(MODEL_NAME_PARAPHRASE_MINILM, cache_folder='./model_cache')
    qdrant_manager = QdrantManager(CONTAINER_URL, CONTAINER_PORT, embedding_model)
    generator = ResponseGenerator('llama3.2:3b-instruct-fp16')

    # process_course_slides(qdrant_manager_instance=qdrant_manager)

    # this will take 2 hours to create a full collection in qdrant
    # process_word_paradigm_dictionary(qdrant_manager_instance=qdrant_manager)

    # qdrant_search_loop(qdrant_manager)

    retriever = QdrantRetriever(qdrant_manager, 'article_pages_cosine')
    query = "What is artificial intelligence?"
    retrieved_docs = retriever.get_relevant_documents(query)
    # retrieved_docs = retriever.get_relevant_chunks(query)
        
    answer = generator.generate_response(retrieved_docs,query)
    print(answer.content)
    
    strValidator = generator.validate_response(retrieved_docs, query, answer.content)
    print('\n')
    print(strValidator.content)
    
    validator = json.loads(strValidator.content)
    print(validator)
    isValid = validator["valid"]
    print(isValid)
    
    index = 0
        
    while not isValid and index < 4 :
        try:
            answer = generator.second_time_generate_response(validator["suggestion"])
            # print(answer)
            strValidator = generator.validate_response(retrieved_docs, query, answer.content)
            print('\n')
            print('----------')
            print(answer)
            print('\n')
            print(strValidator.content)
            print('----------')
            print('\n')
        
            validator = json.loads(strValidator.content)
            isValid = validator["valid"]
            index = index + 1
            # print('test '+ str(index))
            # print('\n')
            # print('\n')
        except :
            # print(answer)
            # print('\n')
            # print(strValidator.content)
            print('\n')
            print('fail')
            index = index + 1
    
    print('final answer')
    print(answer)
            
    
    # print('-------------------')
    # print('index '+ str(index))
    # print(validator.content)