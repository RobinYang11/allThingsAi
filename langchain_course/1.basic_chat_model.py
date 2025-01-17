
from langchain_openai import ChatOpenAI


model = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",  
    model="deepseek-chat",
    api_key="sk-cf89516147f14b30a62457d17af6ed98" # type: ignore
)
res =model.invoke("who are you ?")
print(res.content)