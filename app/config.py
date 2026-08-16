from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    app_name: str = "NP Employee Assistant"
    dataset_path: Path = ROOT_DIR / "employee_np.xlsx"
    processed_path: Path = ROOT_DIR / "data" / "processed" / "employees.jsonl"
    artifact_dir: Path = ROOT_DIR / "artifacts"
    top_k: int = 5
    min_relevance_score: float = 0.05
    llm_provider: str = "extractive"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    def model_post_init(self, __context: object) -> None:
        if not self.dataset_path.is_absolute():
            self.dataset_path = ROOT_DIR / self.dataset_path


@lru_cache
def get_settings() -> Settings:
    return Settings()

