


from langchain_openai import ChatOpenAI
from langchain_core.messages import  AIMessage,HumanMessage,SystemMessage

model = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",  
    model="deepseek-chat",
    api_key="sk-cf89516147f14b30a62457d17af6ed98" # type: ignore
)


chat_history = []

system_message = SystemMessage(content="You are a helpful AI assistant")
chat_history.append(system_message)



while True:
    query = input("You:")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))
    print(f"AI:{response}")





