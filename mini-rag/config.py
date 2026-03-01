from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
import os
class LLMModels(str,Enum):
    open_ai = "open_ai"
    ollama = "ollama"
    mock = "mock"

class DBType:
    faiss = "faiss"
    pgvector = "pgvector"

@dataclass(frozen=True)
class AppConfig():
    chunk_size:int
    chunk_overlap:int
    k:int
    llm_model:LLMModels
    llm_model_name:str
    api_key:str
    db_type:DBType
    transformer_model_name:str
    documents_folder:str
    max_doc_num:int

def load_config()->AppConfig:
    load_dotenv()
    chunk_size:int = os.getenv("CHUNK_SIZE",600)
    chunk_overlap: int = os.getenv("CHUNK_OVERLAP",60)
    k:int = os.getenv("K",3)
    llm_model: LLMModels = os.getenv("LLM_MODEL",LLMModels.ollama)
    llm_model_name = os.getenv("LLM_MODEL_NAME","llama3.2:1b")
    api_key = os.getenv("API_KEY","")
    db_type:DBType = os.getenv("DB_TYPE",DBType.faiss)
    transformer_model_name:str = os.getenv("TRANSFORMER_NAME","BAAI/bge-m3")
    documents_folder:str = "rag_docs"
    max_doc_num:int = os.getenv("MAX_DOC_NUM",10)
    config = AppConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        k=k,
        llm_model=llm_model,
        llm_model_name=llm_model_name,
        api_key=api_key,
        db_type=db_type,
        transformer_model_name=transformer_model_name,
        documents_folder=documents_folder,
        max_doc_num=max_doc_num
        )
    return config