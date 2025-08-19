"""
將 (1) 使用者提問 與 (2) 匹配到的藥品 文本合併，輸出成指定格式。

預設路徑：
  1) data/interim/user_input/test_question.txt
  2) data/interim/case_annex/compare_result.txt
  3) data/interim/case_annex/annex_consequent.txt

用法（可選參數覆寫路徑）：
  python src/text_annex.py \
    --question data/interim/user_input/test_question.txt \
    --compare  data/interim/case_annex/compare_result.txt \
    --output   data/interim/case_annex/annex_consequent.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_QUESTION = Path("data/interim/user_input/test_question.txt")
DEFAULT_COMPARE = Path("data/interim/case_annex/compare_result.txt")
DEFAULT_OUTPUT = Path("data/interim/case_annex/annex_consequent.txt")


def read_text(path: Path) -> str:
    """
    安全讀檔：優先以 UTF-8 讀取；若遇到 BOM 或編碼例外，再退而求其次處理。
    回傳純文字內容；若檔案不存在則丟出 FileNotFoundError。
    """
    if not path.is_file():
        raise FileNotFoundError(f"找不到檔案：{path}")

    # 先嘗試一般 UTF-8
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # 再嘗試處理 BOM
        try:
            return path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            # 最後以系統預設編碼嘗試（不建議，但作為容錯）
            return path.read_text()


def compose(question: str, compare: str) -> str:
    """
    依指定格式合併文本。保留原始內文，不多做裁切。
    會確保檔尾有單一換行。
    """
    # 去除尾端多餘換行，避免輸出過多空白
    q = question.rstrip("\n")
    c = compare.rstrip("\n")

    merged = (
        "一、使用者提問\n"
        f"{q}\n\n"
        "二、匹配到的藥品\n"
        f"{c}\n"
    )
    return merged


def ensure_parent_dir(path: Path) -> None:
    """確保輸出檔案父層目錄存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    """以 UTF-8 無 BOM 寫出內容。"""
    ensure_parent_dir(path)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="合併提問與匹配結果，輸出成 annex_consequent.txt")
    p.add_argument("--question", type=Path, default=DEFAULT_QUESTION, help="使用者提問檔案路徑")
    p.add_argument("--compare", type=Path, default=DEFAULT_COMPARE, help="匹配結果檔案路徑")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="輸出檔案路徑")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        question_text = read_text(args.question)
        compare_text = read_text(args.compare)
    except FileNotFoundError as e:
        print(f"✗ 讀檔失敗：{e}")
        return 1
    except Exception as e:
        print(f"✗ 未預期錯誤（讀檔）：{type(e).__name__}: {e}")
        return 1

    try:
        merged = compose(question_text, compare_text)
        write_text(args.output, merged)
    except Exception as e:
        print(f"✗ 未預期錯誤（寫檔）：{type(e).__name__}: {e}")
        return 1

    print("✓ 合併完成")
    print(f"  問題：{args.question}")
    print(f"  匹配：{args.compare}")
    print(f"  輸出：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
