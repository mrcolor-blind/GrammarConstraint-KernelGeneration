"""
Translate Runner — orchestrates Modal GPU evaluation for translated kernels.
"""

from backends.modal.jobs.translate_evaluation import translate_evaluation
from typing import Optional, Union


class TranslateRunner:
    """Runs evaluation of a generated translation on Modal GPUs."""

    def evaluate(
        self,
        run_id: str,
        concrete_dims:Optional[ dict[str, int] ] = None,
    ) -> dict:
        dims_str = ",".join(f"{k}={v}" for k, v in (concrete_dims or {}).items())
        return translate_evaluation.remote(
            run_id=run_id,
            concrete_dims_str=dims_str,
        )
