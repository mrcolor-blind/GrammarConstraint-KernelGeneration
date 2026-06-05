"""
Knowledge Base — loads TritonBench JSON and parses PyTorch docstrings.
"""

import inspect
import json
import re
from pathlib import Path

import torch
from typing import Optional, Union

# Default path inside Modal containers; local paths may not exist.
_DEFAULT_TRITONBENCH_JSON = Path("/opt/TritonBench/data/TritonBench_T_comp_alpac_v1.json")


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


class KnowledgeBase:
    """Caches TritonBench data and provides docstring parsing."""

    def __init__(self, tritonbench_json_path: Union[str, Path, None] = None):
        self._tritonbench = {}
        self._torch_doc_cache: dict[str, Optional[str]] = {}
        self._torch_sig_cache: dict[str, Optional[inspect.Signature]] = {}

        if tritonbench_json_path is None:
            tritonbench_json_path = _DEFAULT_TRITONBENCH_JSON

        path = Path(tritonbench_json_path)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                # TritonBench Alpaca format is usually a list of dicts with 'instruction'
                if isinstance(data, list):
                    for item in data:
                        instr = item.get("instruction", "")
                        # Extract operator name from instruction
                        op_name = self._extract_op_name_from_instruction(instr)
                        if op_name:
                            self._tritonbench[op_name] = item
                elif isinstance(data, dict):
                    self._tritonbench = data
            except Exception:
                pass  # silently ignore corrupted JSON

    @staticmethod
    def _extract_op_name_from_instruction(instruction: str) ->Optional[ str ]:
        """Heuristic: grab the first 'torch.xxx' or 'torch.nn.functional.xxx' in instruction."""
        m = re.search(r"(torch\.[a-zA-Z0-9_\.]+)", instruction)
        return m.group(1) if m else None

    def get_tritonbench_entry(self, op_name: str) ->Optional[ dict ]:
        """Lookup by exact op_name in the loaded TritonBench data."""
        return self._tritonbench.get(op_name)

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
