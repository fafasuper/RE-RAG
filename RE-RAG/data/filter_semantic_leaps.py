
import os
import json
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_jaccard_similarity(str1: str, str2: str) -> float:
    if not isinstance(str1, str) or not isinstance(str2, str):
        return 0.0
    set1 = set(str1.strip().lower())
    set2 = set(str2.strip().lower())

    if not set1 and not set2:
        return 1.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0
def load_chip_cdn_json(file_path: str) -> list:
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logging.info("✅ 成功按标准 JSON 数组格式读取数据。")
    except json.JSONDecodeError:
        logging.warning("⚠️ 标准 JSON 解析失败，尝试按行 (JSONL) 解析...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        logging.info("✅ 成功按 JSONL 格式读取数据。")
    except Exception as e:
        logging.error(f"❌ 读取文件失败: {e}")

    return data

def filter_semantic_leaps(input_json_path: str, output_csv_path: str, threshold: float = 0.3):
    if not os.path.exists(input_json_path):
        logging.error(f"❌ 未找到输入文件: {input_json_path}，请检查路径。")
        return

    logging.info(f"📂 正在解析 CHIP-CDN 训练集: {input_json_path}")
    raw_data = load_chip_cdn_json(input_json_path)

    if not raw_data:
        logging.error("❌ 数据集为空或解析失败，程序终止。")
        return

    parsed_records = []
    for item in raw_data:
        clinical_text = item.get('text', '')
        normalized_result = item.get('normalized_result', '')
        if clinical_text and normalized_result:
            parsed_records.append({
                "clinical_text": clinical_text,
                "standard_term": normalized_result
            })
    df = pd.DataFrame(parsed_records)
    logging.info(f"🔍 成功提取 {len(df)} 条有效数据对，开始计算字面相似度 (阈值 < {threshold}) ...")

    df['literal_similarity'] = df.apply(
        lambda row: calculate_jaccard_similarity(row['clinical_text'], row['standard_term']),
        axis=1
    )

    filtered_df = df[df['literal_similarity'] < threshold].copy()
    filtered_df['literal_similarity'] = filtered_df['literal_similarity'].round(4)
    logging.info(f"📊 筛选完毕！原始总样本数: {len(df)} | 语义飞跃样本数: {len(filtered_df)}")
    filtered_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    logging.info(f"💾 成功将 {len(filtered_df)} 条数据导出至: {output_csv_path}")

if __name__ == "__main__":
    INPUT_JSON = "CHIP-CDN/CHIP-CDN_train.json"
    OUTPUT_CSV = "data/semantic_leap_samples.csv"
    filter_semantic_leaps(INPUT_JSON, OUTPUT_CSV, threshold=0.3)