
import pandas as pd
import logging
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelEvaluator:
    def __init__(self, result_csv_path: str):
        self.result_csv_path = result_csv_path
    def _parse_terms(self, text: str, sep: str = '##') -> set:
        if pd.isna(text) or not str(text).strip():
            return set()
        return set([t.strip() for t in str(text).split(sep) if t.strip()])

    def evaluate(self) -> dict:
        try:
            df = pd.read_csv(self.result_csv_path)
        except Exception as e:
            logging.error(f"❌ 读取评测文件失败: {e}")
            return {}

        total_samples = len(df)
        if total_samples == 0:
            logging.warning("⚠️ 评测文件为空！")
            return {}

        valid_samples = 0
        strict_correct = 0
        sample_f1_sum = 0.0

        global_tp = 0
        global_fp = 0
        global_fn = 0

        hit_count = 0
        oracle_f1_sum = 0.0

        for _, row in df.iterrows():
            gold_set = self._parse_terms(row.get('gold_standard', ''), sep='##')
            pred_set = self._parse_terms(row.get('final_normalized_result', ''), sep='##')
            cand_set = self._parse_terms(row.get('retrieved_candidates', ''), sep='|')

            if not gold_set:
                continue

            valid_samples += 1
            if gold_set == pred_set:
                strict_correct += 1
            tp = len(gold_set.intersection(pred_set))
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            sample_f1_sum += f1
            global_tp += tp
            global_fp += fp
            global_fn += fn

            if len(gold_set.intersection(cand_set)) > 0:
                hit_count += 1

            # --- 5. Oracle F1 ---
            oracle_pred_set = gold_set.intersection(cand_set)
            o_tp = len(gold_set.intersection(oracle_pred_set))
            o_fp = len(oracle_pred_set - gold_set)
            o_fn = len(gold_set - oracle_pred_set)

            o_precision = o_tp / (o_tp + o_fp) if (o_tp + o_fp) > 0 else 0.0
            o_recall = o_tp / (o_tp + o_fn) if (o_tp + o_fn) > 0 else 0.0
            o_f1 = (2 * o_precision * o_recall) / (o_precision + o_recall) if (o_precision + o_recall) > 0 else 0.0
            oracle_f1_sum += o_f1

        if valid_samples == 0:
            return {}

        strict_acc = strict_correct / valid_samples
        sample_f1 = sample_f1_sum / valid_samples

        micro_p = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
        micro_r = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r) / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

        hit_at_k = hit_count / valid_samples
        oracle_f1 = oracle_f1_sum / valid_samples

        metrics = {
            "valid_samples": valid_samples,
            "strict_accuracy": round(strict_acc, 4),
            "sample_f1": round(sample_f1, 4),
            "micro_f1": round(micro_f1, 4),
            "hit_at_k": round(hit_at_k, 4),
            "oracle_f1": round(oracle_f1, 4)
        }

        logging.info("=" * 45)
        logging.info(f"📊 评测完成 | 有效样本数: {valid_samples}")
        logging.info("-" * 45)
        logging.info(f"  ➤ Strict Accuracy : {metrics['strict_accuracy']:>7.2%}")
        logging.info(f"  ➤ Sample F1       : {metrics['sample_f1']:>7.2%}")
        logging.info(f"  ➤ Micro F1        : {metrics['micro_f1']:>7.2%}")
        logging.info(f"  ➤ Hit@K (召回上限): {metrics['hit_at_k']:>7.2%}")
        logging.info(f"  ➤ Oracle F1       : {metrics['oracle_f1']:>7.2%}")
        logging.info("=" * 45)

        return metrics

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        evaluator = ModelEvaluator(test_path)
        evaluator.evaluate()
    else:
        print("💡 请提供CSV文件路径。")