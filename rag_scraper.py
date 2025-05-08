# This file does the following: 
# Scrapes a Website:  WebBaseLoader
# Chunks & embeds the text with OllamaEmbeddings.
# Stores / reloads the vectors in a local Chroma DB.
# Answers questions with a RetrievalQA chain that uses an Ollama model as the LLM. (Pull the required model)
#  it persists the vector store so you don’t have to rebuild it every time

#importing Libraries

# Chroma DB is an open-source vector store used for storing and retrieving vector embeddings. 
# Its main use is to save embeddings along with metadata to be used later by large language models.
#  Additionally, it can also be used for semantic search engines over text data.


from langchain_community.document_loaders import WebBaseLoader #scrapping
from langchain.text_splitter import RecursiveCharacterTextSplitter  #chunking
from langchain_community.embeddings import OllamaEmbeddings  #embeds for docs
from langchain_community.vectorstores import Chroma #vector db
from langchain.chat_models import init_chat_model  #LLM model
from langchain.chains import RetrievalQA  # Retreival chain 
import pathlib
import argparse
import textwrap
import sys
from bs4 import BeautifulSoup
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DB_DIR = pathlib.Path("chroma_db")      # persisted store folder
OLLAMA_MODEL = "gemma3:1b"  

def build_or_load_db(url:str):  #crawling the website if the vector store not exists
    """Scrape the site (only once) and return a Chroma vector store."""
    if DB_DIR.exists():
        print("Found existing vector DB – loading it.")
        return Chroma(embedding_function=OllamaEmbeddings(), persist_directory=str(DB_DIR))
    
    #scraping website
    print(f"Crawling {url}")
    docs = WebBaseLoader(url).load()
    print(f"{len(docs)} raw docs scraped.")

    # split long pages into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents=docs)
    print(f"Split into {len(chunks)} chunks of ~{CHUNK_SIZE} chars.")

    #creating embeddings and persisting
    vector_db = Chroma.from_documents(documents=chunks, embedding=OllamaEmbeddings(), persist_directory=str(DB_DIR))
    vector_db.persist()
    print(f"Vector store written to {DB_DIR.absolute()}")
    return vector_db

def interactive_qa(vector_db):  #model defining and giving the retreived data
    llm  = init_chat_model(model = OLLAMA_MODEL, model_provider="ollama")

    #reteival chain creation
    # qa_chain = RetrievalQA.from_chain_type(
    # llm,
    # retriever=vector_db.as_retriever(),
    # return_source_documents=True
    # )

    print("\nAsk me anything about the site – type 'exit' to quit.\n")

    while True:
        try:
            q = input(">>> ").strip()  #capturing query/ question
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
             break
        if not q:
            continue

        SYSTEM = """You are an expert assistant.
        Answer *only* from the context between <context></context>;
        if the answer isn’t there, say “I don't know.”"""
        USER = """<context>\n{context}\n</context>\n\nQuestion: {input}"""

        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("user", USER)])

        combine_docs_chain = create_stuff_documents_chain(llm, prompt)

        rag_chain = create_retrieval_chain(vector_db.as_retriever(), combine_docs_chain) 

        # result = qa_chain({"query": q})
        # question = "What is Task Decomposition?"
        result   = rag_chain.invoke({"input": q})

        # answer = textwrap.fill(result, width=90)
        print(result["answer"])

        # print("\n" + result + "\n")

        #  where the answer came from:
        # for d in result["source_documents"]:
        #     print("  •", d.metadata["source"])

def main():
    #cmd parser
    parser = argparse.ArgumentParser(description="RAG over a single website with LangChain+Ollama")
    parser.add_argument("url", help="Root URL of the site to ingest")  #capturing the args url
    args = parser.parse_args()

    print("the url captured from args: ", args.url)

    vectordb = build_or_load_db(args.url)
    interactive_qa(vectordb)

if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) < 2:
        print("Usage: python rag_web_scraper.py")
        sys.exit(1)
    main()


#running the file:python rag_scraper.py https://docs.langchain.com


