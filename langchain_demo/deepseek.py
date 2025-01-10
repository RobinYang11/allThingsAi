from langchain_openai import ChatOpenAI
import os
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import SecretStr
from langchain.tools import StructuredTool
from typing import Optional


def get_current_weather(location: str, unit: Optional[str] = "celsius") -> str:
    """Get the current weather in a given location"""
    if location.lower() == "beijing":
        return f"The weather in Beijing is 20 degrees {unit}"
    return f"The weather in {location} is 22 degrees {unit}"

weather_tool = StructuredTool.from_function(
    func=get_current_weather,
    name="get_current_weather",
    description="Get the current weather in a given location"
)

api_key = os.getenv("DeepSeekApiKey")
if not api_key:
    raise ValueError("DeepSeekApiKey environment variable is not set")


chat_model = ChatOpenAI(
    api_key=SecretStr(api_key),
    base_url="https://api.deepseek.com/v1",  # Added /v1 to base URL
    model="deepseek-chat"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can check the weather by calling the tool."),
    ("user", "{input}"),
    ("assistant", "I will check the weather for you using the appropriate tool."),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


agent = create_openai_tools_agent(chat_model, [weather_tool], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[weather_tool], max_iterations=3,verbose=True)  # Added max_iterations

response = agent_executor.invoke(
    {"input": "What's the weather like in Beijing?"}
)
print("\nTool calling response:")
print(response["output"])


