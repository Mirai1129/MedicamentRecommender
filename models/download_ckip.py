import time

from sentence_transformers import SentenceTransformer

# 指定模型名稱與自訂快取資料夾
model_name = 'ckiplab/bert-base-chinese'
model_cache_path = 'CKIP'  # ❗自訂的模型儲存資料夾

print(f"🚀 [1/3] 開始載入模型：{model_name}")
t0 = time.time()
model = SentenceTransformer(model_name, trust_remote_code=True, cache_folder=model_cache_path)
print(f"✅ 模型成功載入（耗時 {time.time() - t0:.2f} 秒）")

try:
    sentences = [
        "胃託依錠",
        "胃託依錠是用於治療胃、十二指腸潰瘍、幽門痙攣、胃酸過多、胃炎，劑型為錠劑，屬於須由醫師處方使用的藥品。"
    ]

    print("🚀 [2/3] 開始進行向量嵌入（embedding）...")
    t1 = time.time()

    embeddings = model.encode(sentences)
    print(f"✅ 嵌入完成（耗時 {time.time() - t1:.2f} 秒）")
    print("句子的嵌入：", embeddings)

    print("🔍 [3/3] 顯示嵌入結果摘要：")
    print(f"    ➤ 向量數量：{len(embeddings)}")
    print(f"    ➤ 向量維度：{len(embeddings[0])}")
    print(f"    ➤ 第一筆向量前 5 維：{embeddings[0][:5]}")

except Exception as e:
    print("❌ 發生錯誤：")
    print(e)
