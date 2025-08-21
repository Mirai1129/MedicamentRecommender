from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # === 基本設定 ===
    app_name: str = "Pharmacist AI"
    version: str = "2.0.0"
    debug: bool = False
    log_level: str = "INFO"

    # === 路徑設定 ===
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def config_dir(self) -> Path:
        return self.project_root / "config"

    @property
    def prompts_dir(self) -> Path:
        return self.config_dir / "prompts"

    # === 模型設定 ===
    ckip_model_name: str = "ckiplab/bert-base-chinese"
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimension: int = 1536

    # === LLM 設定 ===
    llm_provider: str = "openai"  # openai, anthropic, local
    llm_model_name: str = "gpt-4"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2000

    # === 向量資料庫設定 ===
    vector_index_type: str = "faiss"  # faiss, chroma, pinecone
    similarity_threshold: float = 0.7
    max_retrieval_results: int = 10

    # === API 設定 ===
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # === 快取設定 ===
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600

    # === Pipeline 設定 ===
    max_concurrent_steps: int = 3
    step_timeout_seconds: int = 300

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# 常用模板常數
class PromptNames:
    """提示詞模板名稱常數"""
    CASE_EXTRACTION = "case_extraction"
    PHARMACIST = "pharmacist"
    QUESTION_ANALYSIS = "question_analysis"
    KNOWLEDGE_SUMMARY = "knowledge_summary"

setting = Settings()
print(setting.project_root)

# .env.example
"""
環境變數範例檔案
複製此檔案為 .env 並填入實際值
"""

"""
# === 基本設定 ===
APP_NAME=Pharmacist AI
VERSION=2.0.0
DEBUG=false
LOG_LEVEL=INFO

# === API 金鑰 ===
OPENAI_API_KEY=sk-your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# === 模型設定 ===
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2000

EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIMENSION=1536

CKIP_MODEL_NAME=ckiplab/bert-base-chinese

# === 向量資料庫設定 ===
VECTOR_INDEX_TYPE=faiss
SIMILARITY_THRESHOLD=0.7
MAX_RETRIEVAL_RESULTS=10

# === 快取設定 ===
ENABLE_CACHE=true
CACHE_TTL_SECONDS=3600

# === Pipeline 設定 ===
MAX_CONCURRENT_STEPS=3
STEP_TIMEOUT_SECONDS=300
"""
