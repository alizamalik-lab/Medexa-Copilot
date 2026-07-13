from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_provider: str = "groq"  # "openai" | "anthropic" | "gemini" | "groq"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    groq_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    gemini_model: str = "gemini-1.5-flash"
    groq_model: str = "llama-3.3-70b-versatile"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    similarity_threshold: float | None = None
    max_history_exchanges: int = 15

    data_dir: Path = Path("./data")
    chroma_persist_dir: Path = Path("./chroma_db")
    collection_name: str = "medical_billing_kb"

    @property
    def json_dir(self) -> Path:
        return self.data_dir / "json"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdf"


settings = Settings()