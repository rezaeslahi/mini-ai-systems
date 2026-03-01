from config import AppConfig, load_config
from seed import load_doc_paths
from transformer import Transformer
from retriever import Retrieve, RetrieveResponse
from generator import Generator

def proccess_question(
        transformer:Transformer,
        retriever:Retrieve,
        generator:Generator,
        config:AppConfig,
        question:str)->str:
    retr_resp = retriever.retrieve_top_k_chunck(transformer=transformer,k=config.k,query=question)
    answer = generator.generate_answer(question=question,retrieve_response=retr_resp)
    return answer
    
    

def main():
    config:AppConfig = load_config()
    paths = load_doc_paths(config=config)
    transformer:Transformer = Transformer(config=config,paths=paths)
    retriever:Retrieve = Retrieve(transformer=transformer)
    generator:Generator = Generator(config=config)

    exit_commands = {"exit", "quit", "stop"}

    print("\n######################\n")

    while True:
        question:str = input("\nAsk question:\n").strip()

        if question in exit_commands:
            print("Nice to chat with you today!!")
            break
        answer = proccess_question(
            transformer=transformer,
            retriever=retriever,
            generator=generator,
            config=config,
            question=question)
        print(f"\n Answer:\n {answer}\n######################\n")

if __name__ == "__main__":
    main()
