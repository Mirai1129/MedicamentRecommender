from typing import Dict

from ..config import Settings

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
