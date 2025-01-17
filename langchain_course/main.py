


from langchain.llms.base import LLM
from typing import Optional, List
import requests

class BaiduQianfanLLM(LLM):
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    @property
    def _llm_type(self) -> str:
        return "baidu_qianfan"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "prompt": prompt,
            "max_tokens": 256,  # 可根据需要调整
            "temperature": 0.7
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise ValueError(f"API Error: {response.text}")
        result = response.json()
        return result.get("text", "")  # 根据实际返回格式解析

# 示例使用
api_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro"  # 替换为真实的千帆接口地址
api_key = "0hj85hlDzDiVYMihwbwHzsXB"
llm = BaiduQianfanLLM(api_url=api_url, api_key=api_key)

# 测试调用
output = llm("用一句话解释量子力学")
print(output)