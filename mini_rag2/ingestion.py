from mini_rag2.transformer import Transformer
from mini_rag2.chunk_store import Chunk,ChunkStore
from mini_rag2.retriever import VectorStore
from pathlib import Path
import pdfplumber



def ingest_pdf_in_folder(
        transformer:Transformer,
        chunk_store: ChunkStore,
        vector_store:VectorStore,
        folder_path:str,
        num:int = None):
    allowed_suffix = {".pdf"}
    folder = Path(folder_path)
    paths = [p for p in folder.rglob("*") if p.suffix.lower in allowed_suffix]
    if num is None:
        num = len(paths)
    for p in paths[0:num]:
        text = read_pdf_file(str(p))
        chunks:list[Chunk] = transformer.make_chunks(text,str(p))
        chunk_store.add_chunks(chunks)
        vectors,ids = transformer.make_embeddings_for_chunks(chunks)
        vector_store.add_vectors(vectors=vectors,ids=ids)


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
