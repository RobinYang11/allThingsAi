
import os 

from langchain.text_splitter import (
    # 基础字符分割器,按字符数量分割文本
    CharacterTextSplitter,
    # 递归字符分割器,先按分隔符分割,再按字符数量分割,更智能
    RecursiveCharacterTextSplitter,
    # 使用sentence-transformers模型的分词器进行分割
    SentenceTransformersTokenTextSplitter,
    # 所有分割器的基类
    TextSplitter,
    # 使用tiktoken等分词器按token数量分割文本
    TokenTextSplitter
)

from langchain_community.document_loaders import TextLoader
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings


embeddings = OllamaEmbeddings(
    base_url="http://127.0.0.1:11434",
    model="bge-m3"
)

current_dir  = os.path.dirname(os.path.abspath(__file__))
file_path  = os.path.join(current_dir,"books","pfdsj.txt")
db_dir = os.path.join(current_dir,"db")

if not os.path.exists(file_path):
    raise FileNotFoundError("file not found!")


loader = TextLoader(file_path,encoding="utf-8")
documents = loader.load()

def create_vector_store(docs,store_name):
    persistent_directory = os.path.join(db_dir,store_name)
    if not os.path.exists(persistent_directory):
        db = Chroma.from_documents(docs,embeddings,persist_directory = persistent_directory)
    else:
        print("Vector store already exists! No need to initialize!")

print("\n ------ Using Character-based Splitting")
char_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs,"chroma_db_char")


print("\n ------ Using Sentence-based Splitting")
char_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs,"chroma_db_sent")

print("\n ------ Using token-based Splitting")
char_splitter = TokenTextSplitter(chunk_size=512,chunk_overlap=0)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs,"chroma_db_token")

print("\n ------ Using 文本递归 Splitting")
char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=1000)
char_docs = char_splitter.split_documents(documents)
create_vector_store(char_docs,"chroma_rec_char")

print("\n ------ Using自定义 Splitting")
class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        return text.split("\n\n")

custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents)
create_vector_store(custom_docs,"chroma_db_coustom")


def query_vector_store(store_name,query):
    persistent_directory = os.path.join(db_dir,store_name)
    if os.path.exists(persistent_directory):
        db= Chroma(
            persist_directory=persistent_directory,
            embedding_function=embeddings
        )
        retriever =db.as_retriever(
            search_type="similarity",
            # search_type="similarity_score_threshold",
            # search_type="vector_similarity",  # 基于向量的相似度查询
            # search_type="exact",  # 精确匹配查询
            # search_type="fuzzy",  # 模��匹配查询
            # search_type="prefix",  # 前缀匹配查询
            # search_type="embedding_distance",  # 基于embedding的距离查询
            # search_type="embedding_similarity",  # 基于embedding的相似度查询
            # search_type="multi_modal",  # 多模态查询
            # search_type="multi_modal_embedding_similarity",  # 多模态基于embedding的相似度查询
            # search_type="multi_modal_similarity",  # 多模态基于相似度查询
            search_kwargs={
                "k": 3,
                        #    "score_threshold":0.8,
                           },
        )
        relevant_docs = retriever.invoke(query) 

        for i ,doc in enumerate(relevant_docs,1):
            print("================================================================ \n")
            if doc.metadata:
                print(doc)
                print(f"Source:{doc.metadata.get('source','Unknow')} \n")

    else:
        print("Vector store not found!")


query = "张春桥在中共中央机关刊物《红旗》杂志上发表了?"

# query_vector_store("chroma_db_char",query)
# query_vector_store("chroma_db_sent",query)
# query_vector_store("chroma_db_token",query)
query_vector_store("chroma_rec_char",query)
# query_vector_store("chroma_db_coustom",query)


























