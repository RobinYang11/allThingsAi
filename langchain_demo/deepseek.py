from langchain_openai import ChatOpenAI
import os
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import SecretStr
from langchain.tools import StructuredTool
from typing import Optional
import math
import matplotlib.pyplot as plt
import numpy as np




def get_current_weather(location: str, unit: Optional[str] = "celsius") -> str:
    """Get the current weather in a given location"""
    if location.lower() == "beijing":
        return f"The weather in Beijing is 20 degrees {unit}"
    return f"The weather in {location} is 22 degrees {unit}"

def get_fuck(name: str) -> str:
    print(f"fuck {name}")
    return f"fuck {name}"

def draw_circle(radius: float = 10.0) -> str:
    """Draw a circle with the given radius"""
    theta = np.linspace(0, 2*np.pi, 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    
    plt.figure(figsize=(8, 8))
    plt.plot(x, y)
    plt.axis('equal')
    plt.title(f'Circle with radius {radius}')
    plt.grid(True)
    plt.show()
    return f"Drew a circle with radius {radius}"

weather_tool = StructuredTool.from_function(
    func=get_current_weather,
    name="get_current_weather",
    description="Get the current weather in a given location"
)

fuck_tool = StructuredTool.from_function(
    func=get_fuck,
    name="get_fuck",
    description="问候某人"
)

circle_tool = StructuredTool.from_function(
    func=draw_circle,
    name="draw_circle",
    description="Draw a circle with the specified radius"
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
    ("system", "你好"),
    ("user", "{input}"),
    ("assistant", "问候某人"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])


fuck_prompt = ChatPromptTemplate.from_messages([
    ("system", "你好"),
    ("user", "{input}"),
    ("assistant", "问候某人"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])



agent = create_openai_tools_agent(chat_model, [weather_tool,fuck_tool, circle_tool],prompt)
agent_executor = AgentExecutor(agent=agent, tools=[weather_tool,fuck_tool, circle_tool], max_iterations=3,verbose=True)  # Added max_iterations

response = agent_executor.invoke(
    {"input": "问候李雷,北京天气这么样? 再帮我画一个圆"}
)

print("\nTool calling response:")
print(response["output"])


