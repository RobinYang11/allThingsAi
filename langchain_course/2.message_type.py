
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage ,AIMessage,SystemMessage

model = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",  
    model="deepseek-chat",
    api_key="sk-cf89516147f14b30a62457d17af6ed98" # type: ignore
)


messages = [
    AIMessage(content="robin"),
    SystemMessage(content="test"),
    HumanMessage(content="hello"),
]

res =model.invoke(messages)
print(res.content)