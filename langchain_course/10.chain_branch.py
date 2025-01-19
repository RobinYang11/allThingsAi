from common_model import model
from langchain.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableBranch

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a thank you note for this postive feedback: {feedback}"),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful assistant."),
    ("human", "Generate a response addressing this negative feedback: {feedback}"),
  ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful assistant."),
    ("human", "Generate a request for more details for this neutrual feedback: {feedback}"),
  ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful assistant."),
    ("human", "Generate a message to escalate this feedback to a human agent: {feedback}"),
  ]
)

classification_template = ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful assistant."),
    ("human", "Classify the sentiment of this feedback as positive ,negative,neutrual ,or escalte: {feedback}"),
  ]
)

branchs = RunnableBranch(
  (
    lambda x : "positive" in x,
    positive_feedback_template | model | StrOutputParser()
  ),
  (
    lambda x : "negative" in x,
    negative_feedback_template | model | StrOutputParser()
  ),
  (
    lambda x : "neutral" in x,
    neutral_feedback_template | model | StrOutputParser()
  ),
  escalate_feedback_template | model | StrOutputParser()
)

classification_chain = classification_template | model | StrOutputParser()

chain  = classification_template | branchs

review = " The product is excellent . I relly enjoyed using it and found ti helpful."
result = chain.invoke({"feedback", review})
print(result)








