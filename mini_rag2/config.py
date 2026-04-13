from dataclasses import dataclass
from enum import Enum
import os
from dotenv import load_dotenv

class LLMModel(str,Enum):
    open_ai = "open_ai"
    ollama = "ollama"

class TransformerModel():
    open_ai = "open_ai"
    sentence_transformer = "sentence_transformer"

@dataclass(frozen=True)
class Config():
    llm_model:LLMModel
    llm_model_name:str
    transformer_model:TransformerModel
    transformer_model_name:str
    api_key:str

    chunk_size:int
    chunk_overlap:int
    embedding_dim:int

    top_k:int

def load_app_config()->Config:
    load_dotenv()
    llm_model:LLMModel = os.getenv("LLM_MODEL",LLMModel.ollama)
    llm_model_name:str = os.getenv("LLM_MODEL_NAME","llama3.2:1b")
    transformer_model:TransformerModel = os.getenv("TRANSFORMER_MODEL",TransformerModel.sentence_transformer)
    transformer_model_name:str = os.getenv("TRANSFORMER_MODEL_NAME","BAAI/bge-m3")

    api_key:str = os.getenv("API_KEY","")

    chunk_size:int = os.getenv("CHUNK_SIZE",600)
    chunk_overlap:int = os.getenv("CHUNK_OVERLAP",60)
    top_k:int = os.getenv("TOP_K",5)

    config = Config(
        llm_model=llm_model,
        llm_model_name=llm_model_name,
        transformer_model=transformer_model,
        transformer_model_name=transformer_model_name,
        api_key=api_key,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k
        )
    return config

