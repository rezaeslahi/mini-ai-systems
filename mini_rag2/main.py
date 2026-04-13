from mini_rag2.config import Config,load_app_config
from mini_rag2.chunk_store import ChunkStore
from mini_rag2.transformer import Transformer
from mini_rag2.retriever import VectorStore, retrive_context
from mini_rag2.generator import Generator
from mini_rag2.ingestion import ingest_pdf_in_folder
from pathlib import Path

def run_pipeline():
    config:Config = load_app_config()
    chunk_store:ChunkStore = ChunkStore()
    transformer:Transformer = Transformer(config=config)
    vector_store: VectorStore = VectorStore(cosine=True,config=config)
    generator:Generator = Generator(config=config)
    pdf_folder = Path.cwd() / "mini_rag2/rag_docs" 
    ingest_pdf_in_folder(
        transformer=transformer,
        chunk_store=chunk_store,
        vector_store=vector_store,
        folder_path=pdf_folder,
        num = 2

    )


    exit_commands = {"exit", "quit", "stop"}
    print("\n#####################\n")

    while True:
        question:str = input("\nAsk your question:\n").strip()
        if question in exit_commands:
            print("\nNice to chat with you!\n")
            break
        answer = process_question(
            config=config,
            question=question,
            transformer=transformer,
            vector_store=vector_store,
            chunk_store=chunk_store,
            generator=generator
        )
    print(f"\n Answer:\n {answer}\n######################\n")

def process_question(
        config:Config,
        question:str,
        transformer:Transformer,
        vector_store: VectorStore,
        chunk_store:ChunkStore,
        generator:Generator        
)->str:
    query_vector = transformer.make_embeddings_for_query(question)
    chunk_ids = vector_store.retrieve_top_k_ids(query_vector,config.top_k)
    context = retrive_context(chunk_ids=chunk_ids,chunk_store=chunk_store)
    answer = generator.generate_answer(question,context)
    return answer

