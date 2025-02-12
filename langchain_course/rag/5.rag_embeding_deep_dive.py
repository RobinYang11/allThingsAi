

import os 

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.document_loaders import TextLoader

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings



current_dir  = os.path.dirname(os.path.abspath(__file__))
file_path  = os.path.join(current_dir,"books","pfdsj.txt")
db_dir = os.path.join(current_dir,"db")

if not os.path.exists(file_path):
    raise FileNotFoundError("file not found!")


loader = TextLoader(file_path,encoding="utf-8")
documents = loader.load()


text_splitter = CharacterTextSplitter(chunk_size= 1000,chunk_overlap = 0)
docs = text_splitter.split_documents(documents)


de


