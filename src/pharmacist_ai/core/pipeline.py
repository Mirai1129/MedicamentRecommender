"""
pipeline.py — 專案整合管線

依序執行：
1) 若向量檔缺失或 --force，執行 vector_conversion
2) 依序執行各步驟：question_extraction, compare, text_annex, llm_pharmacist

此版本使用動態匯入 (importlib) 取代 subprocess，執行效率更高。

參數：
  --dry-run   只列出將執行的步驟，不實際執行
  --force     強制重建向量（忽略現有檔案）
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Dict, Any

# 專案根目錄，從 src 目錄往上一層
PROJECT_ROOT = Path(__file__).resolve().parents[3] # TODO: 這段最後要把父資料夾重新改好
# 將專案根目錄加入 Python 搜尋路徑，以便動態匯入
sys.path.insert(0, str(PROJECT_ROOT))

# 需要檢查的向量檔
VEC_FILES = [
    PROJECT_ROOT / "data/processed/vector/medicine_completion/embeddings.npy",
    PROJECT_ROOT / "data/processed/vector/medicine_completion/ids.jsonl",
    PROJECT_ROOT / "data/processed/vector/medicine_completion/stats.json",
    ]

# 各步驟的配置，使用你提供的字典結構
# 這是管線的「資料」，可以獨立於執行邏輯進行修改
PIPELINE_STEPS: List[Dict[str, str]] = [
    {
        "name": "question_extraction",
        "module": "src.pharmacist_ai.core.question_extraction"
    },
    {
        "name": "compare",
        "module": "src.pharmacist_ai.knowledge.compare"
    },
    {
        "name": "text_annex",
        "module": "src.pharmacist_ai.utils.text_annex"
    },
    {
        "name": "llm_pharmacist",
        "module": "src.pharmacist_ai.core.llm_pharmacist"
    }
]

VECTOR_BUILD_STEP = {
    "name": "vector_conversion",
    "module": "src.knowledge_process.vector_conversion"
}

# ----------------------------- 公用工具 ----------------------------- #

class PipelineError(RuntimeError):
    """自定義管線錯誤類別"""
    pass

def exists_all(paths: Iterable[Path]) -> bool:
    """檢查所有給定路徑是否存在"""
    return all(p.exists() for p in paths)

def print_box(msg: str) -> None:
    """列印美觀的訊息方塊"""
    line = "─" * max(10, len(msg) + 2)
    print(f"\n┌{line}\n│ {msg}\n└{line}\n", flush=True)

def run_step(step: Dict[str, str], dry_run: bool = False) -> None:
    """動態匯入並執行指定步驟的 main 函式"""
    step_name = step["name"]
    module_path = step["module"]

    print_box(f"START {step_name}")
    print(f"Module: {module_path}", flush=True)

    if dry_run:
        print_box(f"DRY-RUN SKIP {step_name}")
        return

    try:
        # 動態匯入模組
        module = importlib.import_module(module_path)
        # 尋找並執行 main 函式
        if hasattr(module, "main") and callable(module.main):
            start = time.time()
            # 傳遞參數給 main 函式，或不傳
            module.main()
            dur = time.time() - start
            print_box(f"OK {step_name} (耗時 {dur:.2f}s)")
        else:
            raise PipelineError(f"模組 '{module_path}' 中找不到可執行的 'main' 函式。")
    except ImportError as e:
        print_box(f"FAIL {step_name}")
        raise PipelineError(f"無法匯入模組 '{module_path}'：{e}")
    except Exception as e:
        print_box(f"FAIL {step_name}")
        raise PipelineError(f"執行 '{step_name}' 步驟時發生錯誤：{e}")

def maybe_run_vector_build(force: bool, dry_run: bool) -> None:
    """檢查向量檔；若缺任一或強制重建，執行向量轉換步驟。"""
    need_build = force or not exists_all(VEC_FILES)
    if need_build:
        missing = [p.relative_to(PROJECT_ROOT) for p in VEC_FILES if not p.exists()]
        if missing:
            print("偵測到以下缺失檔案，將先建立向量：")
            for m in missing:
                print(f"  - {m}")
        else:
            print("--force 已指定，將重新建立向量。")
        run_step(VECTOR_BUILD_STEP, dry_run=dry_run)
    else:
        print("向量檔已齊全，略過向量重建。")

# ----------------------------- 進入點 ----------------------------- #

def parse_args() -> argparse.Namespace:
    """解析指令列參數"""
    p = argparse.ArgumentParser(description="專案整合管線")
    p.add_argument("--dry-run", action="store_true", help="只顯示將執行的步驟，不實際執行")
    p.add_argument("--force", action="store_true", help="強制重建向量（忽略現有檔案）")
    return p.parse_args()

def main() -> int:
    """主函數，執行整個管線"""
    args = parse_args()

    print_box("PIPELINE INIT")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.executable}")
    print(f"Dry-run      : {args.dry_run}")
    print(f"Force build  : {args.force}")

    try:
        # 1) 檢查／執行向量建置
        maybe_run_vector_build(force=args.force, dry_run=args.dry_run)

        # 2) 之後依序執行步驟
        for step in PIPELINE_STEPS:
            run_step(step, dry_run=args.dry_run)

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
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())