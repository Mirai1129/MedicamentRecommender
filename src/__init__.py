import os
from pathlib import Path

from dotenv import load_dotenv

from src.exceptions.Exceptions import MissingEnvironmentVariableError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env() -> tuple[str, str]:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1")

    if not openai_api_key:
        raise print(MissingEnvironmentVariableError("Cannot find API Key! Please set `OPENAI_API_KEY` in `.env` file"))

    return openai_api_key, model


OPENAI_API_KEY, OPENAI_MODEL_NAME = load_env()
