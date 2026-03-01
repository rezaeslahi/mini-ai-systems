from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.chat_models import ChatOpenAI, ChatOllama
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.messages import AIMessage
from config import AppConfig, LLMModels

def build_llm(config:AppConfig)->BaseChatModel|FakeListChatModel:
    llm = None

    if config.llm_model == LLMModels.mock:
        llm:FakeListChatModel = FakeListChatModel(responses=[
            AIMessage(content="This is the first response"),
            AIMessage(content="This is the second response")
        ])
    elif config.llm_model == LLMModels.open_ai:
        llm:BaseChatModel = ChatOpenAI(model=config.llm_model_name,temperature=0.0,api_key=config.api_key)
    elif config.llm_model == LLMModels.ollama:
        llm:BaseChatModel = ChatOllama(model=config.llm_model_name,temperature=0.0)
    else:
        raise ValueError("Unsupported LLM model")
    return llm