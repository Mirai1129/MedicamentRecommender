"""
pipeline.py — 專案整合管線
依序執行：
1) 若向量檔缺失或 --force，執行 src/knowledge_process/vector_conversion.py
2) src/question_extraction.py
3) src/knowledge_process/compare.py
4) src/text_annex.py
5) src/llm_pharmacist.py

參數：
  --dry-run  只列出將執行的步驟，不實際執行
  --force    強制重建向量（忽略現有檔案）
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess as sp
import sys
import time
from pathlib import Path
from typing import Iterable, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 專案根目錄（假設 src/ 為專案子目錄）
PYTHON = sys.executable  # 使用當前解譯器，避免 venv 混用

# 需要檢查的向量檔
VEC_FILES = [
    PROJECT_ROOT / "data/processed/vector/medicine_completion/embeddings.npy",
    PROJECT_ROOT / "data/processed/vector/medicine_completion/ids.jsonl",
    PROJECT_ROOT / "data/processed/vector/medicine_completion/stats.json",
]

# 各步驟腳本路徑（依序）
STEPS: Tuple[Path, ...] = (
    PROJECT_ROOT / "src/question_extraction.py",
    PROJECT_ROOT / "src/knowledge_process/compare.py",
    PROJECT_ROOT / "src/text_annex.py",
    PROJECT_ROOT / "src/llm_pharmacist.py",
)

VECTOR_STEP = PROJECT_ROOT / "src/knowledge_process/vector_conversion.py"

PIPELINE_STEPS = [
    {
        "name": "question_extraction",
        "module": "src.core.question_extraction"
    },
    {
        "name": "compare",
        "module": "src.core."
    }
]


# ----------------------------- 公用工具 ----------------------------- #

class PipelineError(RuntimeError):
    pass


def exists_all(paths: Iterable[Path]) -> bool:
    return all(p.exists() for p in paths)


def print_box(msg: str) -> None:
    line = "─" * max(10, len(msg) + 2)
    print(f"\n┌{line}\n│ {msg}\n└{line}\n", flush=True)


def run_script(script_path: Path, dry_run: bool = False) -> None:
    """以目前 Python 執行指定腳本，串流輸出並回傳錯誤碼給呼叫端。"""
    if not script_path.exists():
        raise PipelineError(f"找不到腳本：{script_path}")

    cmd = [PYTHON, str(script_path)]
    print_box(f"START {script_path.relative_to(PROJECT_ROOT)}")
    print(f"$ {' '.join(cmd)}\n", flush=True)

    if dry_run:
        print_box(f"DRY-RUN SKIP {script_path.name}")
        return

    start = time.time()
    # 使用 Popen 串流 stdout/stderr
    with sp.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=sp.PIPE, stderr=sp.STDOUT, bufsize=1, text=True) as proc:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")  # 逐行即時列印
            ret = proc.wait()
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            proc.wait()
            raise
    dur = time.time() - start

    if ret != 0:
        print_box(f"FAIL {script_path.name} (耗時 {dur:.2f}s)")
        raise PipelineError(f"{script_path.name} 退出碼：{ret}")
    else:
        print_box(f"OK {script_path.name} (耗時 {dur:.2f}s)")


def maybe_run_vector_build(force: bool, dry_run: bool) -> None:
    """檢查向量檔；若缺任一或強制重建，執行向量轉換步驟。"""
    need_build = force or not exists_all(VEC_FILES)

    missing = [p for p in VEC_FILES if not p.exists()]
    if need_build:
        if missing:
            print("偵測到以下缺失檔案，將先建立向量：")
            for m in missing:
                print(f"  - {m.relative_to(PROJECT_ROOT)}")
        else:
            print("--force 已指定，將重新建立向量。")
        run_script(VECTOR_STEP, dry_run=dry_run)
    else:
        print("向量檔已齊全，略過向量重建。")


# ----------------------------- 進入點 ----------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="專案整合管線")
    p.add_argument("--dry-run", action="store_true", help="只顯示將執行的步驟，不實際執行")
    p.add_argument("--force", action="store_true", help="強制重建向量（忽略現有檔案）")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print_box("PIPELINE INIT")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {PYTHON}")
    print(f"Dry-run      : {args.dry_run}")
    print(f"Force build  : {args.force}")

    try:
        # 1) 檢查／執行向量建置
        maybe_run_vector_build(force=args.force, dry_run=args.dry_run)

        # 2) 之後依序執行步驟
        for step in STEPS:
            run_script(step, dry_run=args.dry_run)

        print_box("PIPELINE DONE ✅")
        return 0
    except PipelineError as e:
        print(f"\n[錯誤] {e}", file=sys.stderr)
        print_box("PIPELINE ABORTED ❌")
        return 2
    except KeyboardInterrupt:
        print("\n[中斷] 收到使用者中斷訊號（Ctrl+C）。", file=sys.stderr)
        print_box("PIPELINE ABORTED ⛔")
        return 130
    except Exception as e:
        print(f"\n[未預期例外] {type(e).__name__}: {e}", file=sys.stderr)
        print_box("PIPELINE CRASHED 💥")
        return 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")  # 確保子程序輸出即時
    sys.exit(main())
