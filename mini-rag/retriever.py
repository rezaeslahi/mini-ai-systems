import faiss
import numpy as np
from transformer import Transformer,Chunk
from dataclasses import dataclass
from typing import Dict,List
from pydantic import BaseModel,Field

class RetrieveResponse(BaseModel):
    chunks:List[Chunk] = Field(default_factory=list)



class Retrieve():
    def __init__(self,transformer:Transformer,cosine:bool=True):
        self.faiss_index:faiss.Index = None
        self._make_index(transformer=transformer,cosine=cosine)
    
    def _make_index(self,transformer:Transformer,cosine:bool):

        X:np.ndarray = transformer.vectors
        ids:np.ndarray = transformer.vector_ids
        dim = X.shape[1]
        base_index = faiss.IndexFlatIP(dim) if cosine else faiss.IndexFlatL2(dim)
        self.faiss_index = faiss.IndexIDMap2(base_index)
        self.faiss_index.add_with_ids(X,ids)
    
    def retrieve_top_k_chunck(self,transformer:Transformer,k:int, query:str)->RetrieveResponse:
        vectors = transformer.make_embeddings_for_query(query=query)
        _,retrieved_ids = self.faiss_index.search(vectors,k)
        res = RetrieveResponse()
        for id in retrieved_ids[0].tolist():
            chunk = transformer.chunks[id]
            res.chunks.append(chunk)
        return res