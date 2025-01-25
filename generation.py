from langchain_ollama import ChatOllama


class ResponseGenerator:


    def __init__(self, model_name):
        self.llm = ChatOllama(model=model_name, temperature=0.8)
        self.validatorLlm = ChatOllama(model=model_name, temperature=0.8)

    def generate_response(self, context, question):
        prompt = f"""You are an IT specialist. Use the context below to answer the question concisely.

               Context:
               {context}

               Question:
               {question}

               Answer:"""
        # print(prompt)
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
        print(prompt)
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
        # print(prompt)
        return self.validatorLlm.invoke([prompt])
    
