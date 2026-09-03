
import os
import time
import yaml
import logging
import argparse
import pandas as pd
import json
from rules_engine.engine import MedicalRuleEngine
from retrieval.hybrid_search import HybridRetriever
from models.llm_generator import MedicalLLM
from evaluation.evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MedNormPipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.step_name = self.config.get('step_name', 'unknown_step')
        self.is_baseline = self.config.get('is_baseline', False)
        self.enable_rewrite = self.config.get('enable_rewrite', False)
        self.enable_retrieval = self.config.get('enable_retrieval', False)
        self.enable_normalize = self.config.get('enable_normalize', False)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("artifacts", "runs", self.step_name, timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        self._init_modules()
    def _init_modules(self):
        logging.info(f"🚀 正在初始化 {self.step_name} 流水线组件...")

        self.llm = MedicalLLM(
            model_name=self.config.get('llm_model_name', 'gpt-3.5-turbo'),
            temperature=self.config.get('llm_temperature', 0.0)
        )

        if self.enable_rewrite or self.enable_normalize:
            self.rule_engine = MedicalRuleEngine(rules_path=self.config.get('rules_path', 'data/medical_rules_samples.json'))

        if self.enable_retrieval:
            self.retriever = HybridRetriever(
                icd_lib_path=self.config.get('icd_lib_path', 'data/icd_lib_samples.json'),
                faiss_index_path=self.config.get('faiss_index_path', 'artifacts/indexes/faiss/index.faiss')
            )
    def run(self):
        input_data_path = self.config.get('input_data_path', 'data/chip-cdn/test_samples.csv')

        if not os.path.exists(input_data_path):
            logging.error(f"❌ 找不到输入数据集: {input_data_path}")
            return

        logging.info(f"📂 开始处理数据集: {input_data_path}")
        df = pd.read_csv(input_data_path)

        results = []
        for index, row in df.iterrows():
            raw_text = str(row.get('clinical_text', '')).strip()
            gold_standard = str(row.get('standard_term', '')).strip()

            if not raw_text:
                continue
            current_text = raw_text
            candidates = []
            if self.enable_rewrite:
                current_text = self.rule_engine.rewrite(current_text)

            if self.enable_retrieval:
                top_k = self.config.get('retrieval_top_k', 5)
                candidates = self.retriever.retrieve(current_text, top_k=top_k)
                candidates_names = [cand['name'] for cand in candidates]
            else:
                candidates_names = []

            llm_output = self.llm.generate(
                raw_text=current_text,
                candidates=candidates_names,
                is_baseline=self.is_baseline
            )
            final_result = llm_output

            if self.enable_normalize:
                final_result = self.rule_engine.normalize(final_result)

            results.append({
                "original_text": raw_text,
                "gold_standard": gold_standard,
                "rewritten_text": current_text if self.enable_rewrite else "",
                "retrieved_candidates": "|".join(candidates_names),
                "llm_raw_output": llm_output,
                "final_normalized_result": final_result
            })

            if (index + 1) % 50 == 0:
                logging.info(f"⏳ 已处理 {index + 1} 条数据...")

        predictions_path = os.path.join(self.output_dir, "predictions.csv")
        result_df = pd.DataFrame(results)
        result_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')
        logging.info(f"✅ 推理完成！预测结果已保存至: {predictions_path}")

        self._evaluate_and_save(predictions_path)

    def _evaluate_and_save(self, predictions_path: str):
        logging.info("📊 正在计算多维评测指标...")
        evaluator = ModelEvaluator(result_csv_path=predictions_path)
        metrics = evaluator.evaluate()  # 需要微调 evaluator.py 使其返回字典

        if metrics:
            metrics_path = os.path.join(self.output_dir, "metrics.json")
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=4, ensure_ascii=False)
            logging.info(f"🏆 指标计算完成！报告已归档至: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RE-RAG 口")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    args = parser.parse_args()
    pipeline = MedNormPipeline(config_path=args.config)
    pipeline.run()