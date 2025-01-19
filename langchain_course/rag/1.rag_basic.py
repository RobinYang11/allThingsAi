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
        """批量处理文档并返回嵌入向量"""
        # 发送请求到 DeepSeq API 获取文本的嵌入向量
        response = requests.post(self.api_url, json={"texts": texts}, headers={"Authorization": f"Bearer {self.api_key}"})
        embeddings = response.json().get("embeddings", [])
        return embeddings

    def embed_query(self, query: str) -> list:
        """处理查询文本并返回嵌入向量"""
        response = requests.post(self.api_url, json={"texts": [query]}, headers={"Authorization": f"Bearer {self.api_key}"})
        embeddings = response.json().get("embeddings", [])[0]
        return embeddings


deepSeekEmbeddings = DeepSeqEmbeddings(
    api_url=os.getenv("BASE_URL"), # type: ignore
    api_key=os.getenv("API_KEY") # type: ignore
) 

current_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(current_dir, "books", "pfdsj.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist.please check the path"
        )

    # 创建文本加载器，用于读取文本文件
    loader = TextLoader(file_path, encoding="utf-8")
    # 加载文档内容
    documents = loader.load()
    # 创建文本分割器，将文档分割成更小的块
    # chunk_size: 每个文本块的大小
    # chunk_overlap: 相邻文本块之间的重叠部分
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, length_function=len
    )
    # 将文档分割成多个文本块
    docs = text_splitter.split_documents(documents)

    # 使用Chroma向量数据库存储文档
    # 将文档向量化并持久化保存到指定目录
    db = Chroma.from_documents(
        docs, deepSeekEmbeddings, persist_directory=persistent_directory
    )
    
else:
    print("Vector store already exists. No need to initialize.")

