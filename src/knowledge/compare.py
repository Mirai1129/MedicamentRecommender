"""
compare.py — 以關鍵詞陣列（extract_result.json）進行向量比對，輸出相似藥物與介紹至文字檔。

輸入：
  • 關鍵詞檔：data/interim/case_annex/extract_result.json
  • 向量：data/processed/vector/embeddings.npy
  • 向量對應：data/processed/vector/ids.jsonl  （每行: {row, id, preview}）
  • 原始內容：data/raw/medicine_completion.jsonl（逐行 JSONL）

輸出：
  • data/interim/case_annex/compare_result.txt

說明：
  • 相似度採 cosine，相依於 embeddings.npy 已 L2 normalize（對照 vector_conversion.py 的輸出）。
  • 預設 top_k=10、threshold=0.35；可依需求調整 CLI 參數。
  • 會盡量從原始 JSONL 還原對應文字。若找不到常見鍵，會輸出整行 JSON 作為 fallback。

用法：
  python src/knowledge_process/compare.py \
    --queries data/interim/case_annex/extract_result.json \
    --embeddings data/processed/vector/medicine_completion/embeddings.npy \
    --ids data/processed/vector/medicine_completion/ids.jsonl \
    --raw data/raw/medicine_completion.jsonl \
    --out data/interim/case_annex/compare_result.txt \
    --model-dir models/CKIP/models--ckiplab--bert-base-chinese/snapshots \
    --top-k 10 --threshold 0.35
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src import PROJECT_ROOT

# 與向量化一致的候選鍵
CANDIDATE_TEXT_KEYS = (
    "text",
    "content",
    "completion",
    "message",
    "doc",
    "document",
    "prompt",
)


def find_snapshot_dir(snapshots_root: Path) -> Path:
    """尋找實際模型快照目錄"""
    if not snapshots_root.exists():
        raise FileNotFoundError(f"Snapshots root not found: {snapshots_root}")
    if (snapshots_root / "config.json").exists():
        return snapshots_root
    for p in sorted(snapshots_root.iterdir()):
        if p.is_dir() and (p / "config.json").exists():
            return p
    raise FileNotFoundError(
        f"No snapshot subdir with config.json under: {snapshots_root}"
    )


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling to get document embeddings"""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize tensor"""
    return x / (x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps))


def load_ids(ids_path: Path) -> List[Dict]:
    """載入 ids.jsonl 檔案"""
    metas: List[Dict] = []
    with ids_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                metas.append(json.loads(line))
            except json.JSONDecodeError:
                metas.append({"row": None, "id": None, "preview": line})
    return metas


def build_row_lookup(raw_path: Path) -> Dict[int, Dict]:
    """將原始 JSONL 以行號建立查找表（行號以 0 起算，需與 ids.jsonl 中的 row 對齊）。"""
    mapping: Dict[int, Dict] = {}
    with raw_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    obj = {"text": str(obj)}
            except json.JSONDecodeError:
                obj = {"text": line}
            mapping[i] = obj
    return mapping


def extract_text(record: Dict) -> str:
    """從記錄中萃取文字"""
    for k in CANDIDATE_TEXT_KEYS:
        if k in record:
            v = record[k]
            return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return json.dumps(record, ensure_ascii=False)


def embed_queries(queries: List[str], model_dir: Path, device: torch.device, max_length: int = 128) -> np.ndarray:
    """嵌入查詢關鍵字"""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, add_pooling_layer=False).to(device)
    model.eval()

    vecs: List[np.ndarray] = []
    with torch.no_grad():
        for q in queries:
            enc = tokenizer(q, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            out = model(**enc)
            pooled = mean_pooling(out.last_hidden_state, enc["attention_mask"])  # [1, H]
            pooled = l2_normalize(pooled)
            vecs.append(pooled.squeeze(0).cpu().numpy().astype(np.float32))
    if not vecs:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.vstack(vecs)


def search_similar(
        emb_matrix: np.ndarray,  # [N, D] 預期已 L2-normalized
        query_vec: np.ndarray,  # [D] 已 L2-normalized
        top_k: int = 10,
        threshold: float = 0.35,
) -> List[Tuple[int, float]]:
    """搜尋相似的向量"""
    # 餘弦相似度 = 內積（因為已經 L2 normalize）
    sims = emb_matrix @ query_vec  # [N]
    # 先用 argpartition 取前 k，再排序
    k = min(top_k, sims.shape[0])
    if k <= 0:
        return []
    idx = np.argpartition(sims, -k)[-k:]
    idx = idx[np.argsort(-sims[idx])]
    results: List[Tuple[int, float]] = []
    for i in idx:
        s = float(sims[i])
        if s >= threshold:
            results.append((int(i), s))
    return results


def main():
    """主函數，執行比較與輸出"""
    ap = argparse.ArgumentParser(description="Compare query terms to medicine embeddings and export matches.")
    # 修正：所有路徑預設值都改為基於專案根目錄
    ap.add_argument("--queries", type=str,
                    default=str(PROJECT_ROOT / "data/interim/case_annex/extract_result.json"))
    ap.add_argument("--embeddings", type=str,
                    default=str(PROJECT_ROOT / "data/processed/vector/medicine_completion/embeddings.npy"))
    ap.add_argument("--ids", type=str,
                    default=str(PROJECT_ROOT / "data/processed/vector/medicine_completion/ids.jsonl"))
    ap.add_argument("--raw", type=str,
                    default=str(PROJECT_ROOT / "data/raw/medicine_completion.jsonl"))
    ap.add_argument("--out", type=str,
                    default=str(PROJECT_ROOT / "data/interim/case_annex/compare_result.txt"))
    ap.add_argument("--model-dir", type=str,
                    default=str(PROJECT_ROOT / "models/CKIP/models--ckiplab--bert-base-chinese/snapshots"))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.35)
    args = ap.parse_args()

    queries_path = Path(args.queries)
    emb_path = Path(args.embeddings)
    ids_path = Path(args.ids)
    raw_path = Path(args.raw)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 讀 queries
    with queries_path.open("r", encoding="utf-8") as f:
        q_obj = json.load(f)
    if isinstance(q_obj, list):
        queries = [str(x) for x in q_obj if str(x).strip()]
    else:
        raise ValueError("extract_result.json 應為字串陣列。")
    if not queries:
        raise ValueError("未從 extract_result.json 取得任何關鍵詞。")

    # 讀 embeddings（mmap 以節省記憶體）
    emb = np.load(emb_path, mmap_mode="r")  # [N, D] 預期已 L2-normalized

    # 對應資訊 + 原始內容
    metas = load_ids(ids_path)
    if len(metas) != emb.shape[0]:
        raise RuntimeError(f"ids.jsonl 行數({len(metas)}) 與 embeddings 筆數({emb.shape[0]}) 不一致。")
    row_lookup = build_row_lookup(raw_path)

    # 準備模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = find_snapshot_dir(Path(args.model_dir))

    # 嵌入查詢
    q_vecs = embed_queries(queries, model_dir=model_dir, device=device)  # [Q, D]
    # 再確保 L2 normalize（以防不同流程）
    norms = np.linalg.norm(q_vecs, ord=2, axis=1, keepdims=True) + 1e-12
    q_vecs = q_vecs / norms

    # 搜尋與輸出
    buf = io.StringIO()
    global_rank = 1  # 全域編號順號

    for qi, q in enumerate(queries, start=1):
        buf.write(f"=== Query {qi}: {q} ===\n")
        matches = search_similar(emb, q_vecs[qi - 1], top_k=args.top_k, threshold=args.threshold)
        if not matches:
            buf.write("(無符合門檻的結果)\n\n")
            continue
        for idx, score in matches:
            meta = metas[idx] if 0 <= idx < len(metas) else {"row": None, "id": None, "preview": ""}
            row = meta.get("row")
            rid = meta.get("id")
            preview = meta.get("preview") or ""
            full_text = None
            if isinstance(row, int) and row in row_lookup:
                full_text = extract_text(row_lookup[row])

            buf.write(f"[{global_rank}] score={score:.4f}\n")
            if rid is not None:
                buf.write(f"ID: {rid}\n")
            buf.write(f"Preview: {preview}\n")
            if full_text and full_text != preview:
                # 控制長度，避免過長
                show = full_text if len(full_text) <= 2000 else (full_text[:2000] + "…")
                buf.write(f"Text: {show}\n")
            buf.write("\n")
            global_rank += 1
        buf.write("\n")

    with out_path.open("w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    print(f"✓ Exported compare results to: {out_path}")


if __name__ == "__main__":
    main()
