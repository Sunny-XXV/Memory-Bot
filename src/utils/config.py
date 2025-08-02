from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class MemoryAPIConfig(BaseModel):
    base_url: str = Field(..., description="Base URL for the Memory API")
    timeout: int = Field(30, description="Request timeout in seconds")

    @property
    def ingest_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/v1/ingest"

    @property
    def retrieve_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/v1/retrieve"

    def item_url(self, item_id: str) -> str:
        return f"{self.base_url.rstrip('/')}/api/v1/items/{item_id}"

    def task_url(self, task_id: str) -> str:
        return f"{self.base_url.rstrip('/')}/api/v1/tasks/{task_id}"


class MinIOConfig(BaseModel):
    """MinIO configuration for binary storage."""

    endpoint: str = Field(..., description="MinIO server endpoint")
    access_key: str = Field(..., description="MinIO access key")
    secret_key: str = Field(..., description="MinIO secret key")
    bucket_name: str = Field(..., description="MinIO bucket name")
    secure: bool = Field(False, description="Use HTTPS for MinIO connections")


class BotConfig(BaseSettings):
    """Bot configuration from environment variables."""

    telegram_bot_token: str = Field(..., description="Telegram bot token")

    # Memory API settings
    memory_api_base_url: str = Field(..., description="Base URL for the Memory API")
    memory_api_timeout: int = Field(30, description="Memory API timeout in seconds")

    # MinIO settings
    minio_endpoint: str = Field(..., description="MinIO server endpoint")
    minio_access_key: str = Field(..., description="MinIO access key")
    minio_secret_key: str = Field(..., description="MinIO secret key")
    minio_bucket_name: str = Field(..., description="MinIO bucket name")
    minio_secure: bool = Field(False, description="Use HTTPS for MinIO connections")

    # Bot settings
    bot_username: Optional[str] = Field(None, description="Bot username for mention detection")
    enable_logging: bool = Field(True, description="Enable detailed logging")
    log_level: str = Field("DEBUG", description="Logging level")

    # AI Model settings
    llm_model: str = Field("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", description="LLM model name")
    asr_model: str = Field("mistralai/Voxtral-Mini-3B-2507", description="ASR model name")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def memory_api(self) -> MemoryAPIConfig:
        return MemoryAPIConfig(base_url=self.memory_api_base_url, timeout=self.memory_api_timeout)

    @property
    def minio(self) -> MinIOConfig:
        return MinIOConfig(
            endpoint=self.minio_endpoint,
            access_key=self.minio_access_key,
            secret_key=self.minio_secret_key,
            bucket_name=self.minio_bucket_name,
            secure=self.minio_secure,
        )
