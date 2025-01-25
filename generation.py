import os

from langchain_ollama import ChatOllama
import time
import numpy as np
from sentence_transformers import SentenceTransformer

from constants import MODEL_NAME_ALL_MINILM, CONTAINER_URL, CONTAINER_PORT
from qdrant_manager import QdrantManager
from qdrant_retrieval import QdrantRetriever
from questions import  test_question


class ResponseGenerator:

    def __init__(self, model_name, temperature: float = 0.85):
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.validatorLlm = ChatOllama(model=model_name, temperature=temperature)

    def generate_response(self, context, question):
        prompt = f"""You are an assistant. Use the context below to answer the question concisely.
                    If unsure, say "I don’t know based on the given information."

               Context:
               {context}

               Question:
               {question}

               Answer:"""
        return self.llm.invoke([prompt])
    def second_time_generate_response(self, extra_response,):
        extra = ""
        if extra_response:
            extra = f"""
            It is a suggestion from validator about what should be improved in the previous answer, please use this advice, but remember about context and question:
               {extra_response}"""
        prompt = f"""
               {extra}
               
               Improve only the answer and return only improve answer, without any additional texts

               Answer:"""
        # print(prompt)
        return self.llm.invoke([prompt])
    
    def validate_response(self, context, question, answer):
        prompt = f"""
                Is the answer is correct base on the context ?
               Context:
               {context}

               Question:
               {question}

               Answer:
               {answer}
               -----
               the answer should be only json with two fields:
               - valid - boolean
               - suggestion - how to improve answer and what is wrong in the answer
               """
        print(prompt)
        return self.validatorLlm.invoke([prompt])


def generating_model_test():
    models = [
        "huihui_ai/Hermes-3-Llama-3.2-abliterated:3b",
        "deepseek-r1:8b",
        "falcon:7b-instruct",
        "llama3.2:3b-instruct-fp16"
    ]
    temperatures = [0.0, 0.5, 0.9]
    sentence_transformer_model = SentenceTransformer(MODEL_NAME_ALL_MINILM)
    qdrant_manager = QdrantManager(CONTAINER_URL, CONTAINER_PORT, sentence_transformer_model)
    retriever = QdrantRetriever(qdrant_manager, 'article_pages_cosine')
    query = "What is artificial intelligence?"
    documents_retrieved = retriever.get_relevant_documents(query)

    output_file = "models_results.txt"

    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, "w") as f:
        for model in models:
            for temp in temperatures:
                responses_time = []
                for question in  test_question:
                    generator = ResponseGenerator(model_name=model, temperature=temp)

                    try:
                        start_time = time.time()
                        response = generator.generate_response(documents_retrieved, question)
                        response_time = time.time() - start_time
                        responses_time.append(response_time)
                        f.write(f"Model: {model}, Temperature: {temp}\n")
                        f.write(f"Response Time: {response_time:.2f} seconds\n")
                        f.write(f"Question: {question}\n")
                        f.write(f"Response: {response}\n")


                        print(f"Model: {model}, Temp: {temp}, Time: {response_time:.2f}s Response: {response.content}")
                    except Exception as e:
                        f.write(f"Model: {model}, Temperature: {temp} - Error: {str(e)}\n")
                        f.write("=" * 80 + "\n")
                        print(f"Error testing model {model} at temperature {temp}: {e}")
                f.write(f"Model: {model}, Temp: {temp} - Median_time: {np.median(responses_time):.2f}s Mean time: {np.mean(responses_time)}\n")
                f.write("\n")
                f.write("=" * 80 + "\n")
        print(f"Testy zakończone. Wyniki zapisano w pliku: {output_file}")


if __name__ == "__main__":
    generating_model_test()
