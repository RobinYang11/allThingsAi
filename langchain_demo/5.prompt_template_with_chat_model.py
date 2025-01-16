
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    base_url="https://api.deepseek.com/v1",  
    model="deepseek-chat",
    api_key="sk-cf89516147f14b30a62457d17af6ed98" # type: ignore
)

template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)
prompt = prompt_template.invoke({"topic":"dog"})

result = model.invoke(prompt)
print(result.content)



template_multiple = """
    You are helpful assistant.
    Human: Tell me a {adjective} short story about a . {animal}.
    Asssistant:
"""
prompt_template_m = ChatPromptTemplate.from_template(template_multiple)

promptm = prompt_template_m.invoke({
  "adjective":"funny"  ,
  "animal":"dog"
})

result1 = model.invoke(promptm)
print(result1.content)