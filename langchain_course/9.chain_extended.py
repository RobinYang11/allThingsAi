from common_model import model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableLambda

def count_answer(x):
  l = len(x)
  return "there are {l} letters in the answer".format(l=l)

template = "write a story about the {name}"
prompt_template = ChatPromptTemplate.from_template(template)

output_format =  RunnableLambda(lambda x: x.upper())
count_result = RunnableLambda(count_answer)

chain = prompt_template | model | StrOutputParser() | output_format | count_result
result = chain.invoke({"name": "cat"})

print(result)

