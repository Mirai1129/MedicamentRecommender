# config/model_config.py
"""
模型相關配置
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .Settings import Settings

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