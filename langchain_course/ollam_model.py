from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = "写一个关于 {name} 的故事"
prompt_template = ChatPromptTemplate.from_template(template)

embeddings = OllamaEmbeddings()
llm =OllamaLLM(base_url="127.0.0.1:11434",model="deepseek-r1:7b")

chain = prompt_template | llm | StrOutputParser()

res =chain.invoke({"name": "狗"}) 
print(res)
