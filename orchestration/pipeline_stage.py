"""
Orchestration — PipelineStage ABC with 3-attempt repair loop.
Every stage in the translation pipeline inherits from this class.
"""

import logging
from abc import ABC, abstractmethod

from models.domain import PipelineContext, StageResult

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """Abstract base for a repairable pipeline stage."""

    name: str = ""
    max_attempts: int = 3

    def run(self, ctx: PipelineContext) -> StageResult:
        """Execute the stage with up to *max_attempts* repair loops."""
        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"[{self.name}] attempt {attempt}/{self.max_attempts}")
            result = self._try(ctx)

            if result.success:
                logger.info(f"[{self.name}] passed on attempt {attempt}")
                ctx.attempt_counts[self.name] = attempt
                return result

            logger.warning(
                f"[{self.name}] attempt {attempt}/{self.max_attempts} failed: {result.error}"
            )

            if attempt < self.max_attempts:
                ctx = self._prepare_retry(ctx, result.error, attempt)

        logger.error(f"[{self.name}] exhausted all {self.max_attempts} attempts")
        ctx.attempt_counts[self.name] = self.max_attempts
        return result

    @abstractmethod
    def _try(self, ctx: PipelineContext) -> StageResult:
        """Single attempt. Must be implemented by concrete stages."""
        ...

    def _prepare_retry(self, ctx: PipelineContext, error: str, attempt: int) -> PipelineContext:
        """
        Mutate *ctx* before the next attempt.
        Default: no-op.  Subclasses may override to inject error context,
        relax constraints, or switch to fallback strategies.
        """
        return ctx
