# config/settings.py
"""
應用程式配置管理
使用 Pydantic Settings 進行類型安全的配置管理
"""

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """應用程式主要配置"""

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


# config/model_config.py
"""
模型相關配置
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .settings import Settings

settings = Settings()


@dataclass
class ModelConfig:
    """模型配置基類"""
    name: str
    model_path: Optional[Path] = None
    parameters: Dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


@dataclass
class CKIPConfig(ModelConfig):
    """CKIP 模型配置"""

    def __init__(self):
        super().__init__(
            name="ckip",
            model_path=settings.models_dir / "ckip",
            parameters={
                "model_name": settings.ckip_model_name,
                "device": "auto",  # auto, cpu, cuda
                "batch_size": 32,
                "max_length": 512,
            }
        )


@dataclass
class EmbeddingConfig(ModelConfig):
    """嵌入模型配置"""

    def __init__(self):
        super().__init__(
            name="embedding",
            parameters={
                "model_name": settings.embedding_model,
                "dimension": settings.embedding_dimension,
                "batch_size": 100,
                "normalize": True,
            }
        )


@dataclass
class LLMConfig(ModelConfig):
    """LLM 配置"""

    def __init__(self):
        super().__init__(
            name="llm",
            parameters={
                "provider": settings.llm_provider,
                "model_name": settings.llm_model_name,
                "temperature": settings.llm_temperature,
                "max_tokens": settings.llm_max_tokens,
                "top_p": 0.9,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            }
        )


# config/prompts/templates.py
"""
提示詞模板管理
"""

from pathlib import Path
from typing import Dict

from ..settings import Settings

settings = Settings()


class PromptTemplates:
    """提示詞模板管理器"""

    def __init__(self):
        self.prompts_dir = settings.prompts_dir
        self._templates: Dict[str, str] = {}
        self._load_templates()

    def _load_templates(self):
        """載入所有提示詞模板"""
        if not self.prompts_dir.exists():
            return

        for prompt_file in self.prompts_dir.glob("*.txt"):
            template_name = prompt_file.stem
            try:
                content = prompt_file.read_text(encoding='utf-8')
                self._templates[template_name] = content
            except Exception as e:
                print(f"Warning: Failed to load prompt template {prompt_file}: {e}")

    def get(self, template_name: str) -> str:
        """取得提示詞模板"""
        if template_name not in self._templates:
            raise KeyError(f"Prompt template '{template_name}' not found")
        return self._templates[template_name]

    def format(self, template_name: str, **kwargs) -> str:
        """格式化提示詞模板"""
        template = self.get(template_name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required parameter {e} for template '{template_name}'")

    def list_templates(self) -> list[str]:
        """列出所有可用的模板"""
        return list(self._templates.keys())


# 全域實例
prompt_templates = PromptTemplates()


# 常用模板常數
class PromptNames:
    """提示詞模板名稱常數"""
    CASE_EXTRACTION = "case_extraction"
    PHARMACIST = "pharmacist"
    QUESTION_ANALYSIS = "question_analysis"
    KNOWLEDGE_SUMMARY = "knowledge_summary"


# src/pharmacist_ai/utils/logging_utils.py
"""
日誌工具
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config.settings import Settings

settings = Settings()


def setup_logging(
        level: Optional[str] = None,
        log_file: Optional[Path] = None,
        format_string: Optional[str] = None
) -> logging.Logger:
    """設定日誌系統"""

    if level is None:
        level = settings.log_level

    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # 設定根日誌器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # 清除現有處理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # 控制台處理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # 檔案處理器（如果指定）
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """取得命名的日誌器"""
    logger = logging.getLogger(name)

    # 如果沒有設定處理器，則設定基本日誌
    if not logger.handlers and not logger.parent.handlers:
        setup_logging()

    return logger


# 初始化日誌系統
setup_logging()


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