"""
Shape Extraction package — captures exact tensor shapes from call site execution.
"""

from shape_extraction.executor import (
    extract_shapes,
    format_shapes_for_comparison,
    format_shapes_for_prompt,
)

__all__ = [
    "extract_shapes",
    "format_shapes_for_comparison",
    "format_shapes_for_prompt",
]
