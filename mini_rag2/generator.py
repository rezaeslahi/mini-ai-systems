from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models import ChatOpenAI,ChatOllama
from langchain_core.messages import HumanMessage,AIMessage
from mini_rag2.config import LLMModel,Config

class Generator():

    def __init__(self,config:Config):
        self.llm:BaseChatModel
        self.config = config
        self._build_llm_model()

    def _build_llm_model(self):
        model_name = self.config.llm_model_name
        if self.config.llm_model == LLMModel.ollama:
            self.llm = ChatOllama(model=model_name)
        elif self.config.llm_model == LLMModel.open_ai:
            self.llm = ChatOpenAI(model=model_name,api_key=self.config.api_key,temperature=0.0)
        else:
            raise ValueError("Unsupported LLM model")
    
    def generate_answer(self,query:str,retrieve_response:str)->str:
        prompt = self.get_prompt_context(query,retrieve_response)
        msg:AIMessage = self.llm.invoke([HumanMessage(content=prompt)])
        answer = msg.content.strip()
        return answer
        


    
    def get_prompt_context(self, question:str, retrieve_response:str)->str:
        prompt = f"""
You are a factual QA system.

Your task is to answer the user's question ONLY using the provided retrieved chunks.
Do not use any external knowledge.
Do not invent information.
If the answer is not supported by the retrieved text, say:
"Insufficient information in retrieved documents."

User question:
{question}

Retrieved information:
{retrieve_response}

Instructions:

1. Read the chunks carefully.
2. Extract only information that directly answers the question.
3. Write a concise factual answer in paragraph form.
4. After each factual statement, include the source file in parentheses.
   Format: (source: filename.pdf)
5. Do NOT use bullet points.
6. Do NOT add explanations beyond what is supported.
7. Do NOT mention chunk_id.
8. At the end, provide a short 1–2 sentence synthesis labeled exactly as:

Final Answer:

The final answer should summarize the evidence-based conclusion.
"""

        return prompt