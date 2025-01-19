

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