


from openai import OpenAI

import os 

api_key = os.getenv("DeepSeekR1Key")
if not api_key:
    raise ValueError("DeepSeekApiKey environment variable is not set")

openai_api_base = "https://api.siliconflow.cn/v1"

client = OpenAI(
    api_key=api_key,
    base_url=openai_api_base,
)

messages = [{"role": "user", "content": "9.11和9.8 谁大？"}]
response = client.chat.completions.create(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", messages=messages) # type: ignore

reasoning_content = response.choices[0].message.reasoning_content # type: ignore
content = response.choices[0].message.content

print("reasoning_content:", reasoning_content)
print("content:", content)