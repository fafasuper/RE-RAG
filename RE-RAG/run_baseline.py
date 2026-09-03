
import os
import pandas as pd
import logging
from models.llm_generator import MedicalLLM

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class BaselinePipeline:
    def __init__(self):
        logging.info("🚀 正在启动基线 (Baseline) 评估流水线...")
        self.llm = MedicalLLM()

    def process_dataset(self, input_csv_path: str, output_csv_path: str):
        if not os.path.exists(input_csv_path):
            logging.error(f"❌ 找不到输入数据集: {input_csv_path}")
            return

        logging.info(f"📂 开始处理真实数据集 (Baseline 模式): {input_csv_path}")
        df = pd.read_csv(input_csv_path)

        results = []
        for index, row in df.iterrows():
            raw_text = str(row.get('clinical_text', '')).strip()
            gold_standard = str(row.get('standard_term', '')).strip()

            if not raw_text:
                continue

            llm_output = self.llm.generate(
                raw_text=raw_text,
                candidates=[],
                is_baseline=True
            )
            results.append({
                "original_text": raw_text,
                "gold_standard": gold_standard,
                "rewritten_text": raw_text,  # 基线无改写阶段
                "retrieved_candidates": "",  # 基线无检索阶段
                "llm_raw_output": llm_output,
                "final_normalized_result": llm_output  # 基线无规则后处理阶段
            })

            if (index + 1) % 100 == 0:
                logging.info(f"⏳ 已处理 {index + 1} 条真实数据...")

        result_df = pd.DataFrame(results)
        result_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"✅ Baseline 全流程处理完毕！结果已保存在: {output_csv_path}")


if __name__ == "__main__":
    INPUT_FILE = "data/CHIP-CDN/CHIP-CDN_dev.json" #同样根据不同实验调整不同数据集
    BASELINE_OUTPUT_FILE = "data/baseline_output_results.csv"
    pipeline = BaselinePipeline()
    pipeline.process_dataset(INPUT_FILE, BASELINE_OUTPUT_FILE)