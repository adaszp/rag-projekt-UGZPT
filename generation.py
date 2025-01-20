from langchain_ollama import ChatOllama


class ResponseGenerator:


    def __init__(self, model_name):
        self.llm = ChatOllama(model=model_name, temperature=0)

    def generate_response(self, context, question):
        prompt = f"""You are an assistant. Use the context below to answer the question concisely.

               Context:
               {context}

               Question:
               {question}

               Answer:"""
        print(prompt)
        return self.llm.invoke([prompt])
