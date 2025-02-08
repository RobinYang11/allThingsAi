
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
import os
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import SecretStr
from langchain.tools import StructuredTool

api_key = os.getenv("DeepSeekApiKey")
if not api_key:
    raise ValueError("DeepSeekApiKey environment variable is not set")

# chat_model = ChatOpenAI(
#     api_key=SecretStr(api_key), # 换成你的key
#     base_url="https://api.deepseek.com/v1", 
#     model="deepseek-chat",)

chat_model =OllamaLLM(base_url="127.0.0.1:11434",model="deepseek-r1:7b")

your_prompt = ChatPromptTemplate.from_messages([
    ("system", "你好"),
    ("user", "{input}"),
    ("assistant", "问候某人"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

def your_function():
    print("your tool")

your_tool= StructuredTool.from_function(
  func=your_function,
  name="your_function",
  description="从知识库搜索"
)

agent = create_openai_tools_agent(chat_model, [],your_prompt)
agent_executor = AgentExecutor(agent=agent, tools=[your_tool], max_iterations=3,verbose=True)  

response = agent_executor.invoke(
    {"input": "帮我从知识库搜索一下某东西"}
)
