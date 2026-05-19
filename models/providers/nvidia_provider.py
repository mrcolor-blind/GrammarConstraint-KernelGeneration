import os

from openai import OpenAI

from models.interfaces.base_provider import BaseProvider


class NvidiaProvider(BaseProvider):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["NVIDIA_API_KEY"],
            base_url="https://integrate.api.nvidia.com/v1",
        )

    def generate(
        self,
        messages: list[dict],
        model: str,
    ) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=8192,
        )

        return response.choices[0].message.content