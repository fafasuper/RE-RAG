# -*- coding: utf-8 -*-
"""
文件：retrieval/hybrid_search.py
功能：双路混合检索核心。
优化：
  1. 废弃现场动态编码，改为直接读取预构建的读取 index.faiss，极大提升启动速度。
  2. 接口变更为 retrieve，完美对接 main_pipeline.py。
"""

import os
import json
import logging
import jieba
import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

try:
    import faiss
    from sentence_transformers import SentenceTransformer

    HAS_DENSE_ENV = True
except ImportError:
    HAS_DENSE_ENV = False
    logging.warning("⚠️ 未检测到 faiss 或 sentence_transformers 环境，稠密检索将不可用。")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridRetriever:
    def __init__(self, icd_lib_path: str, faiss_index_path: str, m3e_model_path: str = "moka-ai/m3e-base",
                 lambda_weight: float = 0.3):
        self.lambda_weight = lambda_weight
        self.m3e_model_path = m3e_model_path

        logging.info(f"🚀 初始化混合检索模块 | 混合权重 λ (BM25) = {self.lambda_weight}")

        try:
            with open(icd_lib_path, 'r', encoding='utf-8') as f:
                self.icd_data = json.load(f)
            self.icd_names = [item['name'] for item in self.icd_data]
            logging.info(f"✅ 成功加载 {len(self.icd_data)} 条 ICD 标准术语。")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ 找不到知识库文件: {icd_lib_path}，请检查路径。")

        logging.info("分词并构建 BM25 稀疏索引...")
        tokenized_corpus = [jieba.lcut(name) for name in self.icd_names]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=1.2, b=0.75)

        self.dense_model = None
        self.faiss_index = None
        self.standard_embeddings = None

        if HAS_DENSE_ENV and os.path.exists(faiss_index_path):
            try:
                logging.info(f"💾 正在从本地读取预构建的 FAISS 索引: {faiss_index_path}")
                self.faiss_index = faiss.read_index(faiss_index_path)

                # 从预建的 IndexFlatIP 索引中直接倒出向量，维持原本的全局混合归一化逻辑
                logging.info("正在从 FAISS 索引中恢复标准向量矩阵...")
                self.standard_embeddings = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)

                logging.info(f"正在加载 M3E 文本编码器: {self.m3e_model_path} ...")
                self.dense_model = SentenceTransformer(self.m3e_model_path)
                logging.info("✅ FAISS 稠密向量环境加载成功。")
            except Exception as e:
                logging.error(f"❌ 读取本地 FAISS 索引或加载 M3E 失败: {e}。将退化为纯 BM25 检索。")
        else:
            logging.warning(f"⚠️ 未找到本地 FAISS 索引文件 {faiss_index_path}，将自动退化为纯 BM25 检索。")

    def _min_max_normalize(self, scores: np.ndarray) -> np.ndarray:
        max_val = np.max(scores)
        min_val = np.min(scores)
        if max_val == min_val:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        num_docs = len(self.icd_names)
        tokenized_query = jieba.lcut(query)
        raw_bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        norm_bm25_scores = self._min_max_normalize(raw_bm25_scores)

        norm_m3e_scores = np.zeros(num_docs)
        if self.dense_model is not None and self.standard_embeddings is not None:
            query_emb = self.dense_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_emb)

            # 计算其余所有向量的余弦相似度
            cos_sim_scores = np.dot(self.standard_embeddings, query_emb[0])
            norm_m3e_scores = self._min_max_normalize(cos_sim_scores)
            actual_lambda = self.lambda_weight
        else:
            # 环境或索引不存在时，退化为纯 BM25
            actual_lambda = 1.0

        # --- 权重的多路融合 ---
        final_scores = (actual_lambda * norm_bm25_scores) + ((1 - actual_lambda) * norm_m3e_scores)
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        candidates = []
        for idx in top_indices:
            candidates.append({
                "code": self.icd_data[idx].get("code", "UNK"),
                "name": self.icd_data[idx]["name"],
                "final_score": float(final_scores[idx]),
                "bm25_score_norm": float(norm_bm25_scores[idx]),
                "m3e_score_norm": float(norm_m3e_scores[idx])
            })

        return candidates


if __name__ == "__main__":
    ICD_LIB = "data/icd/icd_lib.json"
    FAISS_INDEX = "artifacts/indexes/faiss/icd/index.faiss"
    if os.path.exists(ICD_LIB):
        try:
            retriever = HybridRetriever(
                icd_lib_path=ICD_LIB,
                faiss_index_path=FAISS_INDEX,
                lambda_weight=0.3
            )
            test_queries = ["老慢支", "急性阑尾炎伴穿孔"] #测试数据，是否可以正常链接检索库
            for q in test_queries:
                print(f"\n🔍 [自测] 正在检索: '{q}'")
                results = retriever.retrieve(q, top_k=3)
                for i, res in enumerate(results, 1):
                    print(f"   [{i}] {res['name']} (编码: {res['code']}) | 综合得分: {res['final_score']:.4f}")
        except Exception as e:
            print(f"❌ 自测运行异常: {e}")
    else:
        print(f"💡 提示：当前未检测到真实知识库文件 {ICD_LIB}，请确保您的 RAG 数据库已构建。")