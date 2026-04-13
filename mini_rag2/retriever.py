import faiss
import numpy as np
from mini_rag2.config import Config, TransformerModel
from mini_rag2.chunk_store import ChunkStore

class VectorStore():

    def __init__(self,cosine:bool,config:Config):
        self.config = config
        self.dim = self._get_embedding_dim()
        self.cosine = cosine
        self.faiss_index:faiss.Index = None
        self._make_index()
        

    def _make_index(self):
        base_index = None
        if self.cosine:
            base_index = faiss.IndexFlatIP(self.dim)
        else:
            base_index = faiss.IndexFlatL2(self.dim)
        self.faiss_index = faiss.IndexIDMap2(base_index)
    
    def _get_embedding_dim(self)->int:
        dim = 0
        if self.config.transformer_model == TransformerModel.sentence_transformer:
            dim = 1024
        elif self.config.transformer_model == TransformerModel.open_ai:
            dim = 1536
        return dim


    def add_vectors(self,vectors:np.ndarray,ids:list[int]):
        self.faiss_index.add_with_ids(vectors,ids)
    
    def retrieve_top_k_ids(self,vectors,k)->list[int]:
        _,ids_vec = self.faiss_index.search(vectors,k)
        ids = ids_vec[0].tolist()
        return ids


def retrive_context(chunk_ids:list[int],chunk_store:ChunkStore)->str:
    texts:list[str] = [chunk_store.chunks_dict[id].text for id in chunk_ids]
    context = "\n".join(texts)
    return context
