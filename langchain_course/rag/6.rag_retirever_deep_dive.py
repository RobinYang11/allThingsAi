



import os 

# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import CharacterTextSplitter

from langchain_community.document_loaders import TextLoader

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    base_url="http://127.0.0.1:11434",
    model="bge-m3"
)


current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

def query_vector_store(
    store_name,
    query,
    embedding_func,
    search_type,
    search_kwargs
):
    if os.path.exists(db_dir):
        db = Chroma(
            persist_directory=os.path.join(db_dir, store_name),
            embedding_function=embedding_func
        )

        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
        relevant_docs = retriever.invoke(query)
        for i ,doc in enumerate(relevant_docs,1):
            print(doc)


query ="孙少平的父亲是谁？"            

query_vector_store("chroma_db_with_metadata",query,embeddings,"similarity",{"k":3})





