"""Configuration management for BiblioLingo."""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class Config(BaseModel):
    """Configuration settings for BiblioLingo."""

    # OpenAI API
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

    # MongoDB
    mongo_db_url: str = Field(
        default_factory=lambda: os.getenv(
            "MONGO_DB_URL", "mongodb://admin:password@localhost:27017/bibliolingo?authSource=admin"
        )
    )
    db_name: str = Field(default_factory=lambda: os.getenv("DB_NAME", "bibliolingo"))
    collection_name: str = Field(default_factory=lambda: os.getenv("COLLECTION_NAME", "chunks"))

    # Retrieval settings
    default_alpha: float = Field(default_factory=lambda: float(os.getenv("DEFAULT_ALPHA", "0.5")))
    default_top_k: int = Field(default_factory=lambda: int(os.getenv("DEFAULT_TOP_K", "10")))
    rrf_k: int = Field(default_factory=lambda: int(os.getenv("RRF_K", "60")))
    confidence_threshold: float = Field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))
    )

    # LLM settings
    default_llm_model: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "gpt-4o-mini")
    )
    llm_temperature: float = Field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.0"))
    )
    llm_max_tokens: int = Field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "1000")))

    # Logging
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def validate_required(self) -> None:
        """Validate that required configuration values are set."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        if not self.mongo_db_url:
            raise ValueError("MONGO_DB_URL environment variable is required")


# Global config instance
config = Config()
