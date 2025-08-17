"""
llm_pharmacist.py
讀取 system 提示(prompt/pharmacist.txt) 與使用者案例(data/interim/case_annex/annex_consequent.txt)，
呼叫 OpenAI 取得專業藥師建議，輸出到 data/processed/output/llm_final_recommendations.txt
- $ python src/llm_pharmacist.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# OpenAI SDK v1.x
# pip install "openai>=1.0.0"
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

# ---------- 固定路徑 ----------
PROMPT_PATH = Path("prompt/pharmacist.txt")
INPUT_PATH = Path("data/interim/case_annex/annex_consequent.txt")
OUTPUT_PATH = Path("data/processed/output/llm_final_recommendations.txt")


def read_text(path: Path, encoding: str = "utf-8") -> str:
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")
    return path.read_text(encoding=encoding)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_env(key: str, required: bool = True) -> Optional[str]:
    val = os.getenv(key)
    if required and (val is None or val.strip() == ""):
        raise EnvironmentError(f"環境變數缺失：{key}")
    return val


def call_openai(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_input: str,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    timeout_s: int = 60,
    max_retries: int = 3,
) -> str:
    """
    以 Chat Completions 取得單段文字輸出，含基本重試與超時。
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                timeout=timeout_s,  # ← 直接在 create() 帶入 timeout
            )
            content = (resp.choices[0].message.content or "").strip()
            return content
        except (APITimeoutError, RateLimitError, APIError) as e:
            last_err = e
            # 指數退避
            time.sleep(min(2 ** (attempt - 1), 8))
        except Exception:
            # 其他非預期錯誤不重試
            raise
    raise RuntimeError(f"OpenAI 呼叫失敗（已重試 {max_retries} 次）: {last_err}")


def main() -> int:
    # 載入 .env
    load_dotenv()

    # 讀取環境變數
    api_key = get_env("OPENAI_API", required=True)   # 使用 OPENAI_API
    model = get_env("OPENAI_MODEL", required=True)   # e.g. gpt-4.1

    # 準備 OpenAI 客戶端（也可在此統一設定預設 timeout）
    client = OpenAI(api_key=api_key)

    # 讀檔
    system_prompt = read_text(PROMPT_PATH)
    user_input = read_text(INPUT_PATH)

    # 呼叫 LLM
    output_text = call_openai(
        client=client,
        model=model,
        system_prompt=system_prompt,
        user_input=user_input,
        max_tokens=2048,
        temperature=0.2,
        timeout_s=60,
        max_retries=3,
    )

    # 輸出
    ensure_parent_dir(OUTPUT_PATH)
    OUTPUT_PATH.write_text(output_text + "\n", encoding="utf-8")

    print(f"✓ 已產出：{OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FileNotFoundError as e:
        print(f"[檔案錯誤] {e}", file=sys.stderr)
        raise SystemExit(1)
    except EnvironmentError as e:
        print(f"[環境變數錯誤] {e}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as e:
        print(f"[執行失敗] {e}", file=sys.stderr)
        raise SystemExit(1)
