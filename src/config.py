"""Configuration management for Multi-Agent Deep Research system."""
from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LLM API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None)
    GOOGLE_API_KEY: Optional[str] = Field(default=None)

    # Default LLM Provider & Model
    DEFAULT_LLM_PROVIDER: Literal["openai", "anthropic", "google", "mock"] = Field(
        default="openai"
    )
    DEFAULT_MODEL_NAME: str = Field(default="gpt-4o-mini")

    # Search APIs
    TAVILY_API_KEY: Optional[str] = Field(default=None)

    # Workflow parameters
    MAX_RESEARCH_ITERATIONS: int = Field(default=3)
    MAX_SUBTASKS: int = Field(default=4)
    MAX_SEARCH_RESULTS_PER_QUERY: int = Field(default=5)
    SCRAPE_TIMEOUT_SECONDS: int = Field(default=10)
    CRITIC_MIN_SCORE: float = Field(default=80.0)
    ENABLE_REFLECTION_LOOP: bool = Field(default=True)

    def resolve_provider(self) -> str:
        """Auto-detect provider based on available API keys if default is not configured."""
        if self.DEFAULT_LLM_PROVIDER == "mock":
            return "mock"
        if self.DEFAULT_LLM_PROVIDER == "openai" and self.OPENAI_API_KEY:
            return "openai"
        if self.DEFAULT_LLM_PROVIDER == "anthropic" and self.ANTHROPIC_API_KEY:
            return "anthropic"
        if self.DEFAULT_LLM_PROVIDER == "google" and self.GOOGLE_API_KEY:
            return "google"

        # Fallback to any available key
        if self.OPENAI_API_KEY:
            return "openai"
        if self.ANTHROPIC_API_KEY:
            return "anthropic"
        if self.GOOGLE_API_KEY:
            return "google"

        # If no keys, fallback to mock mode
        return "mock"


settings = Settings()
