from dataclasses import dataclass
from itertools import count
from typing import ClassVar


class Chunk():
    _global_id_gen = count(start=0)

    def __init__(self,text:str,doc_path:str):
        self.text = text
        self.doc_path = doc_path
        self.chunk_id = next(self._global_id_gen)
    
    
    

@dataclass
class ChunkStore():
    chunks:list[Chunk] = []
    chunks_dict:dict[int,Chunk] = {}

    def add_chunk(self,chunk:Chunk):
        self.chunks.append(chunk)
        self.chunks_dict[chunk.chunk_id] = chunk
    
    def add_chunks(self,chunks:list[Chunk]):
        for chunk in chunks:
            self.add_chunk(chunk)        