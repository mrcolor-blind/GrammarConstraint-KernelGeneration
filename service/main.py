"""
FastAPI application entrypoint.
"""

import logging

from fastapi import FastAPI

from service.api.routes import router
from service.db.database import engine, Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="PyTorch → Triton Translation Service",
    description="REST API that wraps the GrammarConstraint-KernelGeneration pipeline.",
    version="0.1.0",
)

app.include_router(router)


@app.on_event("startup")
def on_startup():
    """Create database tables on startup."""
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service.main:app", host="0.0.0.0", port=8000, reload=True)
