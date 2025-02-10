from  dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever ,create_retriever_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# from langchain_community.vectorstores import Chroma

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
import os

load_dotenv()

dir  = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(dir,"db","chroma_db_with_metadata")

embeddings = OllamaEmbeddings(
    base_url="http://127.0.0.1:11434",
    model="bge-m3"
)

db = Chroma(persist_directory=persistent_directory,embedding_function=embeddings)


retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# llm = 








    

create_history_aware_retriever()
    