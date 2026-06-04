"""
models/providers/vllm_grammar_provider.py
------------------------------------------
Provider que corre un servidor vLLM LOCAL con XGrammar como backend
de guided decoding.  Úsalo en Modal cuando quieras grammar-constrained
generation sin depender de la NVIDIA API.

Flujo:
  1.  vllm serve <model> --guided-decoding-backend xgrammar
      (levantado en otro proceso / contenedor Modal)
  2.  Este provider se conecta a ese servidor via OpenAI SDK
  3.  Pasa la gramática como extra_body={"guided_grammar": ebnf}

Arrancar el servidor (ejemplo Modal o local):
  vllm serve mistralai/devstral-small-2507 \
       --guided-decoding-backend xgrammar \
       --port 8000

Configurar la variable de entorno:
  VLLM_BASE_URL=http://localhost:8000/v1
"""

import os
from pathlib import Path

from openai import OpenAI

from models.interfaces.base_provider import BaseProvider

# Carga la gramática una sola vez al importar el módulo
_GRAMMAR_PATH = Path(__file__).parent.parent.parent / "grammars" / "triton_kernel.ebnf"
_TRITON_GRAMMAR = _GRAMMAR_PATH.read_text(encoding="utf-8")


class VllmGrammarProvider(BaseProvider):
    """
    Llama a un servidor vLLM con guided_grammar=<triton_kernel.ebnf>.
    El servidor aplica XGrammar internamente en cada paso de decodificación.
    """

    def __init__(self, base_url: str | None = None):
        self.base_url = base_url or os.environ.get(
            "VLLM_BASE_URL", "http://localhost:8000/v1"
        )
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="not-needed",          # vLLM no requiere API key real
        )

    def generate(self, messages: list[dict], model: str) -> str:
        """
        Genera código Triton con la gramática como constraint.

        extra_body keys relevantes de vLLM + XGrammar:
          guided_grammar          → EBNF completo (nuestro caso)
          guided_decoding_backend → fuerza "xgrammar"
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=4096,
            temperature=0.15,
            top_p=0.95,
            extra_body={
                "guided_grammar": _TRITON_GRAMMAR,
                "guided_decoding_backend": "xgrammar",
            },
        )
        return response.choices[0].message.content or ""
