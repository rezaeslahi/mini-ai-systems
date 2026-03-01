from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.messages import AIMessage,HumanMessage
from config import AppConfig,LLMModels
from llm import build_llm
from retriever import RetrieveResponse

class Generator():

    def __init__(self, config:AppConfig):
        self.llm:BaseChatModel|FakeListChatModel
        self._build_llm(config=config)

    def _build_llm(self,config:AppConfig):

        if config.llm_model == LLMModels.mock:
            self.llm:FakeListChatModel = FakeListChatModel(responses=[
                AIMessage(content="This is the first response"),
                AIMessage(content="This is the second response")
            ])
        elif config.llm_model == LLMModels.open_ai:
            self.llm:BaseChatModel = ChatOpenAI(model=config.llm_model_name,temperature=0.0,api_key=config.api_key)
        elif config.llm_model == LLMModels.ollama:
            self.llm:BaseChatModel = ChatOllama(model=config.llm_model_name,temperature=0.0)
        else:
            raise ValueError("Unsupported LLM model")     
        
    def generate_answer(self,question:str,retrieve_response:RetrieveResponse)->str:
        prompt = self.get_prompt_context(question=question,retrieve_response_json=retrieve_response.model_dump_json())
        msg = self.llm.invoke([HumanMessage(content=prompt)])
        return msg.content.strip()
    
    
    
    def get_prompt_context(self, question:str, retrieve_response_json:str)->str:
        prompt = f"""
You are a factual QA system.

Your task is to answer the user's question ONLY using the provided retrieved chunks.
Do not use any external knowledge.
Do not invent information.
If the answer is not supported by the retrieved text, say:
"Insufficient information in retrieved documents."

User question:
{question}

Retrieved information (JSON list of chunks):
{retrieve_response_json}

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

