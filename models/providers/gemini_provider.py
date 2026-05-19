import os

from google import genai

from models.interfaces.base_provider import BaseProvider


class GeminiProvider(BaseProvider):
    def __init__(self):
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )

    def generate(self, messages: list[dict], model: str) -> str:
        contents = []

        for message in messages:
            contents.append(
                f"{message['role']}: {message['content']}"
            )

        response = self.client.models.generate_content(
            model=model,
            contents="\n\n".join(contents),
        )

        return response.text