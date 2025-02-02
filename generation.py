import os

from langchain_ollama import ChatOllama
import time
import numpy as np
from sentence_transformers import SentenceTransformer

from constants import MODEL_NAME_ALL_MINILM, CONTAINER_URL, CONTAINER_PORT
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from questions import  test_question
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class ResponseGenerator:

    def __init__(self, model_name, temperature: float = 0.85):
        self.chat_history = []
        
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.validatorLlm = ChatOllama(model=model_name, temperature=temperature)
        

    def generate_response(self, context, question):
        
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt= ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following article of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \
        Use three sentences maximum and keep the answer concise.\

        {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        
        contextualize_q_chain = contextualize_q_prompt | self.llm | StrOutputParser()
        
        def contextualized_question(input: dict):
            if input.get("chat_history"):
                return contextualize_q_chain
            else:
                return input["question"]
            
        rag_chain = (
            RunnablePassthrough.assign(
                context = contextualized_question 
            )
            | qa_prompt 
            | self.llm
        )
        
        ai_msg = rag_chain.invoke(
            {
                "question": question,
                "chat_history": self.chat_history
            }
        )
        
        self.chat_history.extend(
            [
                HumanMessage(content=question), ai_msg
            ]
        )
        
        return ai_msg
    
    def validate_response(self, context, question, answer):
        prompt = f"""
        You are a validation assistant. You will be given:
        Question:
        {question}
        
        A Context:
        {context}
        
        A Proposed Answer from the RAG system:
        {answer}
        
        Your job is to evaluate whether the Proposed Answer is correct, consistent with the Context, and appropriately addresses the Question. Follow these steps:
        Relevance: Determine if the answer directly addresses the question.
        Consistency & Accuracy: Check if the answer aligns with the facts provided in the Context. If the answer includes information not supported by the Context, or contradicts it, note that as an inconsistency.
        Completeness: Check whether the answer fully covers what the question is asking (no missing key details).
        Verdict: Decide if the Proposed Answer is valid or invalid:
        If valid, respond: “Valid Answer: [brief justification].”
        If invalid, respond: “Invalid Answer: [brief explanation of what is incorrect or incomplete].” Then, if possible, provide the correct or improved answer referencing the Context.
        Be concise but clear in your explanation. Provide references from the Context where relevant to support your assessment.
        """
        
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
