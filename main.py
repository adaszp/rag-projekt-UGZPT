import json

from qdrant_manager import QdrantManager


def qdrant_search_loop(qdrant_manager_instance: QdrantManager, collection_name: str = COLLECTION_NAME):
    while True:
        user_query = str(input("Enter query you want to search in collection (enter q to quit): "))
        if user_query == 'q':
            break

        results = qdrant_manager_instance.search(collection_name, user_query)
        for idx, result in enumerate(results):
            print(f"{idx + 1}. Document id:{result.id}\nScore: {result.score}\nContent: {result.payload['content']}")



if __name__ == "__main__":
    qdrant_manager = QdrantManager(CONTAINER_URL, CONTAINER_PORT, MODEL_NAME)

    # process_course_slides(qdrant_manager_instance=qdrant_manager)

    # this will take 2 hours to create a full collection in qdrant
    # process_word_paradigm_dictionary(qdrant_manager_instance=qdrant_manager)

    qdrant_search_loop(qdrant_manager)