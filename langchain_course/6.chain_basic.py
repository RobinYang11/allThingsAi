
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
model = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",  
    model="deepseek-chat",
    api_key="sk-cf89516147f14b30a62457d17af6ed98" # type: ignore
)

template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

chain =  prompt_template | model | StrOutputParser()
res =chain.invoke({"topic":"dog"})
print(res)




