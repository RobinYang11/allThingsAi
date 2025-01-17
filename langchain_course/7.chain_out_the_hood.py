from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableSequence

from common_model import model

template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)

format_template = RunnableLambda(lambda x:  prompt_template.format(**x))
invoke_model = RunnableLambda(lambda x: format_template.invoke(x.to_messages()))
format_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_template,middle=[model],last=format_output)

res = chain.invoke({"topic":"dog"})

print(res)











