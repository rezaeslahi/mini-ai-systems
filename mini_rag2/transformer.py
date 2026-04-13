from sentence_transformers import SentenceTransformer
from mini_rag2.chunk_store import Chunk
import pdfplumber
import numpy as np
from mini_rag2.config import Config,TransformerModel
import openai

class Transformer():
    def __init__(self,config:Config):
        self.config = config        
    
    def make_chunks(self,text:str,doc_path:str)->list[Chunk]:
        chunks:list[Chunk] = []
        # text:str
        # if is_pdf(path):
        #     text = read_pdf_file(path)
        # else:
        #     text = read_text_file(path)
        chunk_size = self.config.chunk_size
        chunk_overlap = self.config.chunk_overlap
        i = 0
        j = 0
        n = len(text)
        is_finished = False
        while i<n and not is_finished:
            j = min(n,i+chunk_size)
            sub_text = text[i:j]
            chunk = Chunk(sub_text,doc_path)
            chunks.append(chunk)
            if j == n:
                is_finished = True
                break
            i = j - chunk_overlap            
        return chunks
    
    def make_embeddings_for_chunks(self,chunks:list[Chunk])->tuple[np.ndarray,list[int]]:
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        vectors = self._create_embeddings(texts)
        return vectors,ids
    
    def make_embeddings_for_query(self,query:str)->np.ndarray:        
        vector = self._create_embeddings([query])
        return vector
    
    def _create_embeddings(self,texts:list[str])->np.ndarray:
        model_name = self.config.transformer_model_name
        if self.config.transformer_model == TransformerModel.sentence_transformer:
            model = SentenceTransformer(model_name)
            vectors = model.encode(sentences=texts,normalize_embeddings=True) # dim = [nv,1024]
        elif self.config.transformer_model == TransformerModel.open_ai:
            client = openai.OpenAI(api_key=self.config.api_key)
            resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
            vectors = np.array([e.embedding for e in resp.data])  # dim = [nv,1536]
        else:
            vectors = np.ndarray = np.array([])
        return vectors
        
        
        
    



def read_text_file(path:str)->str:
    text:str
    with open(path,"r",encoding="utf-8") as file:
        text = file.read()
    return text

def read_pdf_file(path:str)->str:
    raw:str
    with pdfplumber.open(path) as pdf:
        raw = ""
        for page in pdf.pages:
            raw += page.extract_text()
    return raw

def is_pdf(path:str)->bool:
    return True
