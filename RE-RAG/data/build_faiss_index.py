
import os
import json
import logging
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_offline_index():
    icd_lib_path = "data/icd/icd_lib.json"
    output_dir = "artifacts/indexes/faiss/icd"
    faiss_index_path = os.path.join(output_dir, "index.faiss")
    m3e_model_path = "moka-ai/m3e-base"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(icd_lib_path):
        logging.error(f"❌ 找不到 ICD 知识典: {icd_lib_path}")
        return

    logging.info(f"📂 正在加载 ICD 知识典: {icd_lib_path}")
    with open(icd_lib_path, 'r', encoding='utf-8') as f:
        icd_data = json.load(f)

    icd_names = [item['name'] for item in icd_data]
    logging.info(f"✅ 成功加载 {len(icd_names)} 条术语，准备进行向量化编码...")

    logging.info(f"🤖 正在加载嵌入模型: {m3e_model_path} (初次运行会自动下载，请耐心等待)")
    try:
        model = SentenceTransformer(m3e_model_path)
    except Exception as e:
        logging.error(f"❌ 模型加载失败，请检查网络或路径: {e}")
        return

    logging.info("⏳ 正在将文本映射为稠密向量 (如果术语较多，这里可能需要几分钟)...")
    embeddings = model.encode(icd_names, convert_to_numpy=True, show_progress_bar=True)

    logging.info("⚙️ 正在构建 FAISS 索引...")
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, faiss_index_path)
    logging.info(f"🎉 建库完成！FAISS 索引已成功保存至: {faiss_index_path}")
    logging.info(f"📊 索引维度: {dimension} | 存储向量数: {index.ntotal}")


if __name__ == "__main__":
    build_offline_index()