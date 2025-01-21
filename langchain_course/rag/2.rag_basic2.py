import os

from langchain.text_splitter import CharacterTextSplitter
from pydantic import SecretStr
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import os
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings

load_dotenv()
embeddings = OllamaEmbeddings(
    base_url="http://127.0.0.1:11434",
    model="llama3",
)

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

query = "孙少平是哪里人?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.2},
)

retriever_docs = retriever.invoke(query)

for i, doc in enumerate(retriever_docs):
    print(doc)
    # if doc.metadata:
        # print(doc.metadata.get("source", "Unknown"))
