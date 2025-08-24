"""
1) 讀取系統提示：prompt/case_extraction.txt
2) 讀取使用者輸入：data/interim/user_input/test_question.txt
3) 呼叫 OpenAI Responses API（model 由 .env OPENAI_MODEL 控制，預設 gpt-4.1）
4) 將結果輸出到：data/interim/case_annex/extract_result.json

環境變數（.env）：
- OPENAI_API 或 OPENAI_API_KEY（二擇一皆可）
- OPENAI_MODEL（例如 gpt-4.1）

相依套件：
- python-dotenv
- openai (>=1.0.0 的新 SDK；from openai import OpenAI)
"""
import sys
import time
from pathlib import Path
from typing import Optional

from openai import APIError, APIConnectionError, RateLimitError
from openai import OpenAI
from openai.types.responses import ResponseInputTextParam, Response
from openai.types.responses.response_input_param import Message

from src import PROJECT_ROOT, OPENAI_API_KEY, OPENAI_MODEL_NAME
from src.exceptions.Exceptions import PathNotFoundException

# from __future__ import annotations

# ----------- 常量與路徑設定 -----------
PROMPT_PATH = PROJECT_ROOT / "prompt/case_extraction.txt"
USER_INPUT_PATH = PROJECT_ROOT / "data/interim/user_input/test_question.txt"
OUTPUT_PATH = PROJECT_ROOT / "data/interim/case_annex/extract_result.json"

MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.5


def read_text(path: Path, label: str) -> str:
    """讀取文字檔，若不存在則中止。"""
    if not path.exists():
        raise FileNotFoundError(f"[Error] 找不到 {label} 檔案：{path}")
    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        # 保守做法，若編碼非 UTF-8，嘗試以 cp950 之類再讀一次
        # TODO: encoding 這邊要思考一下怎麼做
        return path.read_text(encoding="utf-8", errors="ignore").strip()


def is_path_exist(path: Path) -> None:
    """確保輸出目錄存在。"""
    if not path.exists():
        raise PathNotFoundException(f"path '{path}' is not found")


def get_openai_responses(client: OpenAI, model: str, system_prompt: str, user_content: str) -> str:
    """
    使用 Responses API 呼叫模型。
    - 將 system_prompt 放在 system role
    - 將 user_content 放在 user role
    回傳純文字輸出。
    """
    # 使用 Responses API（新式）。output_text 會自動將段落合併成純文字。
    resp: Response = client.responses.create(
        model=model,
        input=[
            Message(role="system", content=[ResponseInputTextParam(text=system_prompt, type="input_text")]),
            Message(role="user", content=[ResponseInputTextParam(text=user_content, type="input_text")])
        ],
    )

    # Response 有個 attribute 叫做 output_text，他會 return 一串 string，如果沒有東西就會是空字串
    # return getattr(resp, "output_text").strip()

    try:
        return resp.output_text.strip()
    except AttributeError:
        raise RuntimeError("請升級 OpenAI SDK 到支援 output_text 的版本")


def generate_openai_response_with_retry(client: OpenAI, model: str, system_prompt: str, user_content: str) -> str:
    """帶重試的請求流程，處理網路錯誤與限速。"""
    last_err: Optional[Exception] = None

    for tried_count in range(1, MAX_RETRIES + 1):
        try:
            return get_openai_responses(client, model, system_prompt, user_content)
        except Exception as e:
            last_err = e

            if not _is_retryable_error(error=e):
                # 為了維持基本服務，目前維持 print error 的做法
                # TODO: 修改成 logger 形式
                print(f"[Error] 遭遇無法重試的錯誤 ({type(e).__name__})，停止重試 {e}", file=sys.stderr)
                break

            sleep_s = BACKOFF_BASE_SECONDS ** tried_count
            print(
                f"⚠️ 第 {tried_count}/{MAX_RETRIES} 次遭遇暫時性錯誤 ({type(e).__name__})：{e}\n"
                f"   {sleep_s:.1f}s 後重試…",
                file=sys.stderr,
            )
            time.sleep(sleep_s)

    print("[Error] 產生失敗。最後錯誤：", repr(last_err), file=sys.stderr)
    sys.exit(2)


def _is_retryable_error(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, RateLimitError)):
        return True

    if isinstance(error, APIError):
        try:
            status_code = int(error.code)
            return 500 <= status_code < 600
        except (AttributeError, ValueError, TypeError):
            return False

    return False


def main() -> None:
    system_prompt = read_text(PROMPT_PATH, "系統提示（prompt/case_extraction.txt）")
    user_content = read_text(USER_INPUT_PATH, "使用者輸入（test_question.txt）")

    client = OpenAI(api_key=OPENAI_API_KEY)

    print("🚀 開始呼叫 OpenAI…", file=sys.stderr)
    output_text = generate_openai_response_with_retry(client, OPENAI_MODEL_NAME, system_prompt, user_content)

    if not output_text:
        print("⚠️ 模型未回傳內容（空字串）。", file=sys.stderr)

    is_path_exist(OUTPUT_PATH)
    OUTPUT_PATH.write_text(output_text, encoding="utf-8")
    print(f"✅ 已輸出結果：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
