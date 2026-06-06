"""
Context Resolver — 4-tier fallback for enriching each operator with semantic context.
"""

import inspect

from context.knowledge_base import KnowledgeBase
from models.domain import OpContext, ParamDesc
from typing import Optional, Union


class ContextResolver:
    """Resolves operator context using Tier 1 → Tier 4 fallback."""

    def __init__(self, knowledge_base:Optional[ KnowledgeBase ] = None):
        self.kb = knowledge_base or KnowledgeBase()
        self._cache: dict[str, OpContext] = {}

    def resolve(self, op_name: str) -> OpContext:
        """Resolve an operator name to a rich OpContext."""
        if op_name in self._cache:
            return self._cache[op_name]

        ctx = self._try_resolve(op_name)
        self._cache[op_name] = ctx
        return ctx

    def _try_resolve(self, op_name: str) -> OpContext:
        # --- Tier 1: TritonBench JSON ---
        entry = self.kb.get_tritonbench_entry(op_name)
        if entry:
            instruction = entry.get("instruction", "")
            # Use the instruction itself as the rich description
            return OpContext(
                op_name=op_name,
                source="tritonbench_json",
                confidence="high",
                functional_description=instruction[:500],
                math_formula=self._extract_math_from_text(instruction),
                signature="",
                parameters=[],
                shapes_info=None,
                broadcasting=None,
                edge_cases=None,
                notes="Source: TritonBench JSON (complete variant)",
                full_instruction=instruction,
            )

        # --- Tier 2: torch.__doc__ ---
        doc = self.kb.get_torch_docstring(op_name)
        if doc and len(doc.strip().splitlines()) >= 3:
            parsed = self.kb.parse_docstring(op_name)
            confidence = "medium"
            # If very short docstring, downgrade confidence
            if len(doc.strip().splitlines()) < 5:
                confidence = "low"

            sig = self.kb.get_torch_signature(op_name)
            sig_str = str(sig) if sig else ""
            params = self._signature_to_params(sig) if sig else []

            return OpContext(
                op_name=op_name,
                source="torch_docstring",
                confidence=confidence,
                functional_description=parsed.get("functional_description", ""),
                math_formula=parsed.get("math_formula"),
                signature=sig_str,
                parameters=params,
                shapes_info=parsed.get("shapes_info"),
                broadcasting=parsed.get("broadcasting"),
                edge_cases=parsed.get("edge_cases"),
                notes=None,
            )

        # --- Tier 3: inspect.signature ---
        sig = self.kb.get_torch_signature(op_name)
        if sig:
            params = self._signature_to_params(sig)
            return OpContext(
                op_name=op_name,
                source="inspect_signature",
                confidence="medium",
                functional_description=f"PyTorch built-in: {op_name}",
                math_formula=None,
                signature=str(sig),
                parameters=params,
                shapes_info=None,
                broadcasting=None,
                edge_cases=None,
                notes="Only signature available; no detailed documentation.",
            )

        # --- Tier 4: name only ---
        return OpContext(
            op_name=op_name,
            source="name_only",
            confidence="low",
            functional_description=f"Operator: {op_name}",
            math_formula=None,
            signature="",
            parameters=[],
            shapes_info=None,
            broadcasting=None,
            edge_cases=None,
            notes="No documentation found. LLM must rely on prior knowledge.",
        )

    @staticmethod
    def _extract_math_from_text(text: str) ->Optional[ str ]:
        """Look for LaTeX or math-like expressions in instruction text."""
        import re
        # Find $...$ or $$...$$
        m = re.search(r"\$\$(.+?)\$\$", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"\$(.+?)\$", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        # Find .. math:: style
        m = re.search(r"\.\. math::\s*(.+?)(?:\n\n|\Z)", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return None

    @staticmethod
    def _signature_to_params(sig: inspect.Signature) -> list[ParamDesc]:
        params = []
        for name, p in sig.parameters.items():
            params.append(
                ParamDesc(
                    name=name,
                    type_str=str(p.annotation) if p.annotation != inspect.Parameter.empty else None,
                    default=p.default if p.default != inspect.Parameter.empty else None,
                    required=(p.default == inspect.Parameter.empty and p.kind not in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    )),
                    description="",
                )
            )
        return params
