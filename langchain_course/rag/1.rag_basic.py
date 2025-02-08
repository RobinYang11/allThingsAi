


import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.embeddings.base import Embeddings
import os
from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_deepseek import ChatDeepSeek


embeddings = OllamaEmbeddings(
    base_url="http://127.0.0.1:11434",
    model="bge-m3",
)

load_dotenv()
api_key = os.getenv("DeepSeekR1Key")
if not api_key:
    raise ValueError("DeepSeekApiKey environment variable is not set")

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
        docs, embeddings, persist_directory=persistent_directory
    )
    
else:
    print("Vector store already exists. No need to initialize.")

