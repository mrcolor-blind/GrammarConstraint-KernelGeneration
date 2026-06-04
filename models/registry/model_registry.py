from models.providers.openai_provider import OpenAIProvider
from models.providers.nvidia_provider import NvidiaProvider
from models.providers.gemini_provider import GeminiProvider
from models.providers.vllm_grammar_provider import VllmGrammarProvider


PROVIDERS = {
    "openai":          OpenAIProvider,
    "nvidia":          NvidiaProvider,           # hosted API, sin grammar
    "nvidia-grammar":  NvidiaProvider,           # hosted/NIM + grammar constraint
    "gemini":          GeminiProvider,
    "vllm":            VllmGrammarProvider,      # vLLM local con XGrammar
}


def load_provider(name: str):
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: {name!r}. "
            f"Available: {list(PROVIDERS.keys())}"
        )

    # nvidia-grammar activa grammar_constrained=True
    if name == "nvidia-grammar":
        return NvidiaProvider(grammar_constrained=True)

    return PROVIDERS[name]()
