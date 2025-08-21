"""
vector_conversion.py — 將 data/raw/medicine_completion.jsonl 轉成 BERT 向量（.npy）

• 模型：使用本機 CKIP BERT（HuggingFace 轉換器）
  models/CKIP/models--ckiplab--bert-base-chinese/snapshots/<hash>
• 輸入：data/raw/medicine_completion.jsonl 逐行 JSON（自動偵測欄位）
• 輸出：data/processed/vector/
    - embeddings.npy  (float32) 形狀: [N, D]
    - ids.jsonl        每行對應輸入的 id（若存在）與文字摘要
    - stats.json       維度、筆數、模型路徑、時間等中繼資訊

用法：
    python -m src.knowledge_process.vector_conversion \
        --input data/raw/medicine_completion.jsonl \
        --model-dir models/CKIP/models--ckiplab--bert-base-chinese/snapshots \
        --out-dir data/processed/vector \
        --text-field text  #（可選）指定欄位，否則自動偵測

說明：
- 不依賴 sentence-transformers，直接用 transformers + mean-pooling，避免 S-BERT 專用頭的需求。
- 自動挑選 snapshots 目錄下第一個含 config.json 的子目錄。
- 預設 batch_size=64，可用引數調整；自動偵測 GPU。
- 嚴格 L2 normalize，便於後續 cosine 相似度檢索。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ----------------------------
# Utilities
# ----------------------------

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
    """在 snapshots 根目錄下尋找第一個包含 config.json 的子資料夾。"""
    if not snapshots_root.exists():
        raise FileNotFoundError(f"Snapshots root not found: {snapshots_root}")
    # 直接允許傳入的就是最終目錄（已含 config.json）
    if (snapshots_root / "config.json").exists():
        return snapshots_root
    for p in sorted(snapshots_root.iterdir()):
        if p.is_dir() and (p / "config.json").exists():
            return p
    raise FileNotFoundError(
        f"No snapshot subdir with config.json under: {snapshots_root}"
    )


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """標準 mean-pooling：對非 padding token 做平均。"""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps))


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    obj = {"text": str(obj)}
            except json.JSONDecodeError:
                # 異常行以純文字處理
                obj = {"text": line}
            obj["__row_index__"] = i
            yield obj


def extract_text(record: Dict, preferred_key: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """從單一 JSON 物件中抽出要向量化的文字與可選 id。
    回傳：(text, item_id)
    """
    # 優先取 id-like 欄位（若存在）
    item_id = None
    for k in ("id", "_id", "uid", "uuid", "key"):
        if k in record and isinstance(record[k], (str, int)):
            item_id = str(record[k])
            break

    # 指定欄位優先
    if preferred_key and preferred_key in record:
        value = record[preferred_key]
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return text.strip(), item_id

    # 自動偵測常見鍵
    for k in CANDIDATE_TEXT_KEYS:
        if k in record:
            v = record[k]
            text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            return text.strip(), item_id

    # 後備：將全部鍵值串起（保底不丟資料）
    joined = json.dumps(record, ensure_ascii=False)
    return joined, item_id


# ----------------------------
# Core
# ----------------------------

def embed_corpus(
        texts: List[str],
        tokenizer: AutoTokenizer,
        model: AutoModel,
        device: torch.device,
        batch_size: int = 64,
        max_length: int = 512,
) -> np.ndarray:
    model.eval()
    all_vecs: List[np.ndarray] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i: i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = model(**encoded)
            last_hidden = out.last_hidden_state  # [B, L, H]
            pooled = mean_pooling(last_hidden, encoded["attention_mask"])  # [B, H]
            pooled = l2_normalize(pooled)
            all_vecs.append(pooled.cpu().numpy().astype(np.float32))

    return np.vstack(all_vecs) if all_vecs else np.zeros((0, model.config.hidden_size), dtype=np.float32)


# ----------------------------
# Main CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL to BERT embeddings (CKIP bert-base-chinese)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path("data") / "raw" / "medicine_completion.jsonl"),
        help="輸入 JSONL 路徑",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Path("models") / "CKIP" / "models--ckiplab--bert-base-chinese" / "snapshots"),
        help="CKIP 模型 snapshots 根目錄或最終 snapshot 目錄",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path("data") / "processed" / "vector" / "medicine_completion"),
        help="輸出資料夾路徑",
    )
    parser.add_argument("--text-field", type=str, default=None, help="指定輸入 JSONL 的文字欄位名稱")
    parser.add_argument("--batch-size", type=int, default=64, help="推論批量大小")
    parser.add_argument("--max-length", type=int, default=512, help="Token 最長長度")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 解析模型目錄
    snapshot_root = Path(args.model_dir)
    model_dir = find_snapshot_dir(snapshot_root)

    # 載入模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, add_pooling_layer=False).to(device)
    model.to(device)

    # 讀取 JSONL
    records: List[Dict] = []
    texts: List[str] = []
    ids_meta: List[Dict] = []

    for obj in iter_jsonl(input_path):
        text, item_id = extract_text(obj, preferred_key=args.text_field)
        texts.append(text)
        rec_meta = {
            "row": obj.get("__row_index__"),
            "id": item_id,
            "preview": text[:200],  # 方便檢查
        }
        ids_meta.append(rec_meta)
        records.append(obj)

    if not texts:
        raise RuntimeError(f"No valid texts parsed from: {input_path}")

    # 產生向量
    embeddings = embed_corpus(
        texts,
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # 儲存
    vec_path = out_dir / "embeddings.npy"
    np.save(vec_path, embeddings)

    ids_path = out_dir / "ids.jsonl"
    with ids_path.open("w", encoding="utf-8") as f:
        for m in ids_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    stats = {
        "count": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "device": str(device),
        "model_dir": str(model_dir),
        "input": str(input_path),
        "embeddings_file": str(vec_path),
        "ids_file": str(ids_path),
    }
    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("✓ Done.")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
