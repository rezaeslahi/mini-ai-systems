from dataclasses import dataclass
from config import AppConfig
from sentence_transformers import SentenceTransformer
from typing import List,Dict
import numpy as np
from pydantic import BaseModel,Field
import pdfplumber

class Chunk(BaseModel):
    text:str = Field(...)
    doc_path:str = Field(...)
    chunk_id:int = Field(...)

class Transformer():
    
    def __init__(self,config:AppConfig,paths:List[str]):
        self.model = SentenceTransformer(config.transformer_model_name)
        self.chunks:List[Chunk] = []
        self.vectors:np.ndarray        
        self.vector_ids:np.ndarray
        self.config = config
        self._make_chunks(paths=paths)
        self._make_embeddings()        

    def _make_chunks(self,paths:List[str]):
        chunk_id = 0
        for path in paths:
            raw:str = read_pdf_file(path=path)            
            n = len(raw)
            is_end = False
            i = 0
            while i<n and not is_end:
                j = min(n,i+self.config.chunk_size)
                text = raw[i:j]
                chunk:Chunk = Chunk(text=text,doc_path=path,chunk_id=chunk_id)
                chunk_id += 1
                self.chunks.append(chunk)
                if j== n:
                    is_end = True
                i = j -self.config.chunk_overlap
    

    def _make_embeddings(self):
        texts:List[str] = [chunk.text for chunk in self.chunks]
        vectors:np.ndarray = self.model.encode(sentences=texts, normalize_embeddings=True)
        self.vectors = vectors
        nv = vectors.shape[0] 
        self.vector_ids = np.arange(nv,dtype=np.int64)     

    def make_embeddings_for_query(self,query:str)->np.ndarray:
        vectors:np.ndarray = self.model.encode([query],normalize_embeddings=True)
        return vectors

def read_general_text_file(path:str)->str:
    raw:str
    with open(path,"r",encoding="utf-8") as file:
        raw = file.read()
    return raw

def read_pdf_file(path:str)->str:
    raw:str
    with pdfplumber.open(path) as pdf:
        raw = ""
        for page in pdf.pages:
            raw += page.extract_text()
    return raw
