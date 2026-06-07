"""
Service configuration — env vars, constants, defaults.
"""

from pydantic_settings import BaseSettings


DEFAULT_MODEL = "qwen/qwen3.5-397b-a17b"


class Settings(BaseSettings):
    """Loads env vars with sensible defaults."""

    database_url: str = "sqlite:///data/service.db"
    nvidia_api_key: str | None = None
    default_provider: str = "nvidia"
    default_model: str = DEFAULT_MODEL

    class Config:
        env_prefix = ""  # read plain names like DATABASE_URL
        env_file = ".env"
        extra = "ignore"


settings = Settings()
