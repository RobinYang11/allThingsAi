
import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    base_url=os.getenv("BASE_URL"),  
    model=os.getenv("MODEL_NAME"), # type: ignore
    api_key=os.getenv("API_KEY") # type: ignore
)


