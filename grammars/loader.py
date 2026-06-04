"""
grammars/loader.py
------------------
Carga y compila la gramática Triton-DSL para XGrammar.

Uso mínimo:
    from grammars.loader import load_triton_grammar
    compiled = load_triton_grammar(tokenizer, vocab_size)

Luego pasa `compiled` a:
  - xgr.contrib.hf.LogitsProcessor  (HuggingFace local)
  - vLLM  extra_body guided_grammar  (servidor vLLM)
  - SGLang sampling_params ebnf      (SGLang)
"""

from pathlib import Path

import xgrammar as xgr

# Ruta al archivo EBNF relativa a este módulo
_GRAMMAR_FILE = Path(__file__).parent / "triton_kernel.ebnf"


def load_triton_grammar(
    tokenizer,
    vocab_size: int,
) -> xgr.CompiledGrammar:
    """
    Lee triton_kernel.ebnf, compila la gramática contra el
    vocabulario del tokenizer y devuelve un CompiledGrammar
    listo para usar en GrammarMatcher o LogitsProcessor.

    Parámetros
    ----------
    tokenizer   : tokenizer de HuggingFace (AutoTokenizer)
    vocab_size  : tamaño del vocabulario del modelo
                  (config.vocab_size)

    Retorna
    -------
    xgr.CompiledGrammar
    """
    grammar_str = _GRAMMAR_FILE.read_text(encoding="utf-8")

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=vocab_size,
    )
    compiler = xgr.GrammarCompiler(tokenizer_info)
    return compiler.compile_grammar(grammar_str)


def get_grammar_string() -> str:
    """
    Devuelve el EBNF como string sin compilar.
    Útil para pasarlo como `guided_grammar` en vLLM / SGLang.
    """
    return _GRAMMAR_FILE.read_text(encoding="utf-8")
