
import pandas as pd
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataBuilder:
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def prepare_icd_library(self, excel_path: str, output_filename: str = "icd_lib_samples.json") -> str:
        output_path = os.path.join(self.data_dir, output_filename)
        logging.info(f"正在读取原始 ICD 词表: {excel_path}")

        try:
            df = pd.read_excel(excel_path, header=None)

            if df.shape[1] < 2:
                raise ValueError("Excel文件格式错误，至少需要包含Code（第一列）和Name（第二列）两列数据")

            df.columns = ['code', 'name']
            icd_list = []

            for _, row in df.iterrows():
                code = str(row['code']).strip() if pd.notna(row['code']) else ""
                name = str(row['name']).strip() if pd.notna(row['name']) else ""
                if not code and not name:
                    continue

                icd_list.append({
                    "code": code,
                    "name": name,
                    # search_text 融合了编码和名称，为后续密集检索提供更丰富的上下文
                    "search_text": f"{name} (ICD编码: {code})"
                })
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(icd_list, f, ensure_ascii=False, indent=2)

            logging.info(f"✅ 成功转换 {len(icd_list)} 条标准术语！已保存至: {output_path}")
            return output_path

        except FileNotFoundError:
            logging.error(f"未找到指定的Excel文件，请检查路径：{excel_path}")
            return ""
        except Exception as e:
            logging.error(f"构建知识库失败：{str(e)}")
            return ""

if __name__ == "__main__":
    RAW_EXCEL_PATH = r"E:\pycharm project\chinese_diagnosis_icd\CHIP-CDN\国际疾病分类ICD-10北京临床版v601.xlsx"
    builder = DataBuilder(data_dir=".")
    if os.path.exists(RAW_EXCEL_PATH):
        builder.prepare_icd_library(RAW_EXCEL_PATH)
    else:
        logging.warning(f"本地未找到 {RAW_EXCEL_PATH}。")