from models.providers.openai_provider import OpenAIProvider
from models.providers.nvidia_provider import NvidiaProvider
from models.providers.gemini_provider import GeminiProvider


PROVIDERS = {
    "openai": OpenAIProvider,
    "nvidia": NvidiaProvider,
    "gemini": GeminiProvider,
}


def load_provider(name: str):
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}")

    return PROVIDERS[name]()