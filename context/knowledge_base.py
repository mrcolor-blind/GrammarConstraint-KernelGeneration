"""
Knowledge Base — loads TritonBench JSON and parses PyTorch docstrings.
"""

import inspect
import json
import os
import re
from pathlib import Path

import torch
from typing import Optional, Union


def _parse_docstring(doc: str) -> dict:
    """
    Parse a PyTorch-style docstring into structured fields.
    Looks for: signature line, .. math::, Args:, Keyword args:, etc.
    """
    result = {
        "functional_description": "",
        "math_formula": None,
        "parameters": [],
        "keyword_parameters": [],
        "shapes_info": None,
        "broadcasting": None,
        "edge_cases": None,
        "notes": None,
    }

    if not doc or not doc.strip():
        return result

    lines = doc.strip().splitlines()
    result["functional_description"] = lines[0].strip() if lines else ""

    # Extract .. math:: blocks
    math_blocks = []
    in_math = False
    buffer = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(".. math::"):
            in_math = True
            buffer = []
            continue
        if in_math:
            if stripped == "" or stripped.startswith(".."):
                math_blocks.append("\n".join(buffer))
                in_math = False
                buffer = []
            else:
                buffer.append(line.strip())
    if buffer:
        math_blocks.append("\n".join(buffer))
    if math_blocks:
        result["math_formula"] = math_blocks[0]

    # Simple regex-based extraction for Args and Keyword args
    # We do a naive parse; full RST parsing is overkill for MVP
    doc_text = doc

    # Broadcasting hint
    if "broadcast" in doc_text.lower():
        result["broadcasting"] = "Supports broadcasting to a common shape."

    # Edge cases / notes
    if "nan" in doc_text.lower() or "inf" in doc_text.lower():
        result["edge_cases"] = "Refer to docstring for NaN/Inf behavior."

    return result


def _discover_tritonbench_json() -> Path:
    """Discover the TritonBench JSON file via multiple fallback paths."""
    # 1. Environment variable override
    env_path = os.environ.get("TRITONBENCH_JSON_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # 2. Common local paths (project-relative, home directory)
    candidates = [
        Path(__file__).resolve().parent.parent / "datasets" / "tritonbench" / "TritonBench_T_comp_alpac_v1.json",
        Path.home() / ".tritonbench" / "TritonBench_T_comp_alpac_v1.json",
        Path("/opt/TritonBench/data/TritonBench_T_comp_alpac_v1.json"),
    ]
    for p in candidates:
        if p.exists():
            return p

    # 3. Default (may not exist locally)
    return candidates[-1]


class KnowledgeBase:
    """Caches TritonBench data and provides docstring parsing."""

    def __init__(self, tritonbench_json_path: Union[str, Path, None] = None):
        self._tritonbench: dict[str, dict] = {}
        self._short_name_to_torch: dict[str, str] = {}
        self._torch_doc_cache: dict[str, Optional[str]] = {}
        self._torch_sig_cache: dict[str, Optional[inspect.Signature]] = {}

        if tritonbench_json_path is None:
            tritonbench_json_path = _discover_tritonbench_json()

        path = Path(tritonbench_json_path)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                # TritonBench Alpaca format is usually a list of dicts with 'instruction'
                if isinstance(data, list):
                    for item in data:
                        instr = item.get("instruction", "")
                        # Extract operator name from instruction
                        short_name, op_name = self._extract_op_name_from_instruction(instr)
                        if op_name:
                            self._tritonbench[op_name] = item
                            if short_name and short_name != op_name:
                                self._short_name_to_torch[short_name] = op_name
                elif isinstance(data, dict):
                    self._tritonbench = data
            except Exception:
                pass  # silently ignore corrupted JSON

    @staticmethod
    def _extract_op_name_from_instruction(instruction: str) -> tuple[Optional[str], Optional[str]]:
        """
        Extract the operator name from a TritonBench instruction.
        
        Returns (short_name, canonical_torch_name) where:
        - short_name: the raw function name from the wrapper entry (e.g. "gelu")
        - canonical_torch_name: the fully qualified torch name (e.g. "torch.nn.functional.gelu")
        
        The function uses the "Wrapper Entry Information" line to find the function name,
        then resolves it to the canonical torch module path.
        """
        # Extract the function name from "Wrapper Entry Information: func_name("
        m = re.search(
            r"Wrapper Entry Information:\s*(?:def\s+)?([a-zA-Z0-9_]+)\(",
            instruction,
        )
        if not m:
            # Fallback: search for any torch. path in the instruction
            m2 = re.search(r"(torch\.[a-zA-Z0-9_\.]+)", instruction)
            if m2:
                return None, m2.group(1)
            return None, None
        
        short_name = m.group(1)
        
        # Try to resolve to a canonical torch name
        # Priority: torch.nn.functional > torch > torch.special
        candidates = [
            f"torch.nn.functional.{short_name}",
            f"torch.{short_name}",
            f"torch.special.{short_name}",
            f"torch.linalg.{short_name}",
        ]
        
        for candidate in candidates:
            obj = KnowledgeBase._resolve_torch_obj(candidate)
            if obj is not None:
                return short_name, candidate
        
        # If we can't resolve it, just return the short name as-is
        return short_name, short_name

    def get_tritonbench_entry(self, op_name: str) ->Optional[ dict ]:
        """Lookup by exact op_name in the loaded TritonBench data."""
        # Direct lookup
        if op_name in self._tritonbench:
            return self._tritonbench[op_name]
        
        # Try short name lookup
        if op_name in self._short_name_to_torch:
            canonical = self._short_name_to_torch[op_name]
            return self._tritonbench.get(canonical)
        
        # Try reverse: if op_name is a canonical name, see if we have a short mapping
        short = op_name.split(".")[-1]
        if short in self._short_name_to_torch:
            canonical = self._short_name_to_torch[short]
            return self._tritonbench.get(canonical)
        
        return None

    def get_torch_docstring(self, op_name: str) ->Optional[ str ]:
        """Fetch __doc__ for a torch operator, with caching."""
        if op_name in self._torch_doc_cache:
            return self._torch_doc_cache[op_name]

        doc = None
        try:
            obj = self._resolve_torch_obj(op_name)
            if obj is not None and hasattr(obj, "__doc__"):
                doc = obj.__doc__
        except Exception:
            pass

        self._torch_doc_cache[op_name] = doc
        return doc

    def get_torch_signature(self, op_name: str) -> Optional[inspect.Signature]:
        """Fetch inspect.signature for a torch operator, with caching."""
        if op_name in self._torch_sig_cache:
            return self._torch_sig_cache[op_name]

        sig = None
        try:
            obj = self._resolve_torch_obj(op_name)
            if obj is not None and callable(obj):
                sig = inspect.signature(obj)
        except Exception:
            pass

        self._torch_sig_cache[op_name] = sig
        return sig

    @staticmethod
    def _resolve_torch_obj(op_name: str):
        """Resolve a string like 'torch.add' or 'torch.nn.functional.relu' to the actual object."""
        parts = op_name.split(".")
        if parts[0] != "torch":
            return None
        obj = torch
        for part in parts[1:]:
            if not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        return obj

    def parse_docstring(self, op_name: str) -> dict:
        """Structured parse of the docstring for a given op."""
        doc = self.get_torch_docstring(op_name)
        return _parse_docstring(doc)
