from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = "write a story about the {name}"
prompt_template = ChatPromptTemplate.from_template(template)

embeddings = OllamaEmbeddings()

llm =OllamaLLM(base_url="127.0.0.1:11434",model="deepseek-r1:7b")



chain = prompt_template | llm | StrOutputParser()


res =chain.invoke({"name": "cat"}) 
print(res)
