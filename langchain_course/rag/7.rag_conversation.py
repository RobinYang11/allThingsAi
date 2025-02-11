from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings
import os
from pydantic import SecretStr

load_dotenv()

api_key = os.getenv("DeepSeekR1Key")
if not api_key:
    raise ValueError("DeepSeekApiKey environment variable is not set")

dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(dir, "db", "chroma_db_with_metadata")

embeddings = OllamaEmbeddings(
    base_url="http://127.0.0.1:11434",
    model="bge-m3"
)

llm = ChatOpenAI(
    api_key=SecretStr(api_key),
    base_url="https://api.siliconflow.cn/v1",
    model="deepseek-ai/DeepSeek-V3"
)

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

contextualize_q_system_prompt = """列举一个聊天记录和用户最近的一个问题。
那个是根据历史聊天记录和用户最近的一个问题来生成回复的。
定制一个独立的可以理解的问题。
不要根据历史聊天就录。不要回复用户的问题。
如果有需要可以重新定制一个问题,其他情况直接返回。"""

msg = [
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
]

contextualize_q_prompt = ChatPromptTemplate.from_messages(msg)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """你是一个回答问题的系统,使用检索数据库的内容回答问题。
如果你不知道答案,就不要回答。
使用最大的可能性回答问题。
{context}"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation!")
    chat_history = []
    while True:
        query = input("你: ")
        if query.lower() == "exit":
            break
        result = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        print("AI:", result["answer"])  # Access the answer from the result dictionary
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))

if __name__ == "__main__":
    continual_chat()
