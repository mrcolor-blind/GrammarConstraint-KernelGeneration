"""
TritonBench Registry — lookup de operadores disponibles en el dataset.

Uso:
    from datasets.tritonbench.registry import TritonBenchRegistry
    reg = TritonBenchRegistry()
    reg.is_bench_operator("gelu")      # True
    reg.is_bench_operator("my_custom") # False
    entry = reg.get_entry("gelu")      # dict con instruction/input/output
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

_DATASET_PATH = Path(__file__).parent / "TritonBench_T_comp_alpac_v1.json"

_WRAPPER_RE = re.compile(
    r"Wrapper Entry Information:\s*(?:def\s+)?([a-zA-Z0-9_]+)\("
)


def _extract_operator_name(instruction: str) -> Optional[str]:
    m = _WRAPPER_RE.search(instruction)
    return m.group(1) if m else None


class TritonBenchRegistry:
    """Carga el dataset una sola vez y expone lookups O(1)."""

    def __init__(self, dataset_path: Path = _DATASET_PATH):
        self._entries: dict[str, list[dict]] = {}  # name → lista de entries
        self._load(dataset_path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data:
            name = _extract_operator_name(entry.get("instruction", ""))
            if name:
                self._entries.setdefault(name, []).append(entry)

    # ── API pública ──────────────────────────────────────────────────────────

    def is_bench_operator(self, function_name: str) -> bool:
        """True si el operador existe en TritonBench."""
        return function_name in self._entries

    def get_entry(self, function_name: str) -> Optional[dict]:
        """Devuelve el primer entry del dataset para este operador, o None."""
        entries = self._entries.get(function_name)
        return entries[0] if entries else None

    def operator_names(self) -> set[str]:
        """Conjunto completo de nombres de operadores en el dataset."""
        return set(self._entries.keys())


@lru_cache(maxsize=1)
def get_registry() -> TritonBenchRegistry:
    """Singleton con cache — carga el dataset una sola vez."""
    return TritonBenchRegistry()
