

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# template = "Tell me a joke about {topic}"
# prompt_template = ChatPromptTemplate.from_template(template)

# print("---------------")
# prompt = prompt_template.invoke({"topic":"dog"})
# print(prompt)
# print("--------------")



# template_multiple = """You are a helpful assistant.
# Human: Tell me a joke about {adjective} story about a {animal}.
# Assistant: """

# prompt_template_multiple = ChatPromptTemplate.from_template(template_multiple)

# prompt = prompt_template_multiple.invoke({"adjective":"sad","animal":"cat"})
# print(prompt)



### Prompt with system and human Message
messages = [
   ("system","You are a comedian who tells jokes about {topic}."),
   ("human","Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)

prompt = prompt_template.invoke({"topic":"dog","joke_count":2})
print(prompt)





