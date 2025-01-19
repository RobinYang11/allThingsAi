import os

from langchain.text_splitter import CharacterTextSplitter
from pydantic import SecretStr
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
import requests
import os
from dotenv import load_dotenv

load_dotenv()


class DeepSeqEmbeddings(Embeddings):
    def __init__(self, api_url: str, api_key: str = None):
        self.api_url = api_url
        self.api_key = api_key  # 如果有API Key，可以传递

    def embed_documents(self, texts: list) -> list:
        response = requests.post(
            self.api_url,
            json={"texts": texts},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        embeddings = response.json().get("embeddings", [])
        return embeddings

    def embed_query(self, query: str) -> list:
        response = requests.post(
            self.api_url,
            json={"texts": [query]},
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        embeddings = response.json().get("embeddings", [])
        # Add safety check to handle empty embeddings list
        if not embeddings:
            return []
        return embeddings[0]


deepSeekEmbeddings = DeepSeqEmbeddings(
    api_url=os.getenv("BASE_URL"),  # type: ignore
    api_key=os.getenv("API_KEY"),  # type: ignore
)

current_dir = os.path.dirname(os.path.abspath(__file__))

persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = DeepSeqEmbeddings(
    api_url=os.getenv("BASE_URL"),  # type: ignore
    api_key=os.getenv("API_KEY"),  # type: ignore
)

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)


query = "孙少平是谁？"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.8},
)

retriever_docs = retriever.invoke(query)

for i, doc in enumerate(retriever_docs):
    if doc.metadata:
        print(doc.metadata.get("source", "Unknown"))
