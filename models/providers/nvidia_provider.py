import os

from openai import OpenAI

from models.interfaces.base_provider import BaseProvider


class NvidiaProvider(BaseProvider):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["NVIDIA_API_KEY"],
            base_url="https://integrate.api.nvidia.com/v1",
        )

    def generate(self, messages: list[dict], model: str) -> str:
        """
        Stream the response and separate reasoning tokens (e.g. DeepSeek-R1
        style) from the actual content.  Only the content part is returned so
        the rest of the pipeline stays unchanged.
        """
        stream = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=8192,
            temperature=0.15,
            top_p=0.95,
            seed=42,
            stream=True,
        )

        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # reasoning_content is present in DeepSeek-R1 and similar models
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)

            if delta.content:
                content_parts.append(delta.content)

        return "".join(content_parts)