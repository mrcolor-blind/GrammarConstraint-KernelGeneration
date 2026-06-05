import os
from pathlib import Path

from openai import OpenAI

from models.interfaces.base_provider import BaseProvider
from typing import Optional, Union

# Carga la gramática una sola vez al importar el módulo.
# Path relativo a este archivo → grammars/triton_kernel.ebnf
_GRAMMAR_PATH = Path(__file__).parent.parent.parent / "grammars" / "triton_kernel.ebnf"
_TRITON_GRAMMAR:Optional[ str ] = None

def _load_grammar() -> str:
    global _TRITON_GRAMMAR
    if _TRITON_GRAMMAR is None:
        if _GRAMMAR_PATH.exists():
            _TRITON_GRAMMAR = _GRAMMAR_PATH.read_text(encoding="utf-8")
        else:
            raise FileNotFoundError(f"Grammar file not found: {_GRAMMAR_PATH}")
    return _TRITON_GRAMMAR


class NvidiaProvider(BaseProvider):
    """
    Provider para la NVIDIA Inference API (integrate.api.nvidia.com)
    o para un NIM self-hosted.

    ┌─────────────────────────────────────────────────────────────────┐
    │  grammar_constrained=True  →  usa guided_grammar (XGrammar)    │
    │  grammar_constrained=False →  generación libre (comportamiento  │
    │                               original con stream=True)         │
    └─────────────────────────────────────────────────────────────────┘

    IMPORTANTE — soporte por endpoint:
      • self-hosted NIM (http://tu-nim:8000/v1)
            guided_grammar ✅  confirmado en docs oficiales NIM.
            Úsalo con base_url apuntando a tu NIM en Modal.

      • hosted API (integrate.api.nvidia.com)
            guided_grammar ❓  no documentado para el API cloud.
            Puede que funcione en algunos modelos; prueba con
            grammar_constrained=True y observa si da error 400/422.
            Si falla, usa self-hosted NIM o el VllmGrammarProvider.

    Nota sobre streaming:
      guided_grammar requiere stream=False (el backend XGrammar
      necesita controlar el ciclo completo de decodificación).
      Sin gramática se mantiene stream=True para menor latencia.
    """

    def __init__(
        self,
        base_url:Optional[ str ] = None,
        grammar_constrained: bool = False,
    ):
        self.base_url = base_url or os.environ.get(
            "NVIDIA_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        )
        self.grammar_constrained = grammar_constrained

        self.client = OpenAI(
            api_key=os.environ["NVIDIA_API_KEY"],
            base_url=self.base_url,
        )

    # ------------------------------------------------------------------
    # Generación sin grammar constraint  (comportamiento original)
    # ------------------------------------------------------------------

    def _generate_streaming(self, messages: list[dict], model: str) -> str:
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

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                reasoning_parts.append(reasoning)

            if delta.content:
                content_parts.append(delta.content)

        return "".join(content_parts)

    # ------------------------------------------------------------------
    # Generación CON grammar constraint  (XGrammar via NIM)
    # ------------------------------------------------------------------

    def _generate_grammar_constrained(
        self, messages: list[dict], model: str
    ) -> str:
        """
        Pasa la gramática EBNF como guided_grammar en extra_body.

        XGrammar (backend por defecto en NIM) aplica la máscara en
        cada paso de decodificación del lado del servidor, por lo que
        el output siempre respeta la gramática Triton.

        stream=False es obligatorio: el servidor necesita controlar
        el ciclo completo para aplicar las máscaras de tokens.
        """
        grammar = _load_grammar()

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=8192,
            temperature=0.15,
            top_p=0.95,
            seed=42,
            stream=False,           # requerido para guided decoding
            extra_body={
                "guided_grammar": grammar,
                # "guided_decoding_backend": "xgrammar",  # ya es el default en NIM
            },
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Punto de entrada público (interfaz BaseProvider)
    # ------------------------------------------------------------------

    def generate(self, messages: list[dict], model: str) -> str:
        if self.grammar_constrained:
            return self._generate_grammar_constrained(messages, model)
        return self._generate_streaming(messages, model)
