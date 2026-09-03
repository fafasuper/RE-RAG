
import os
import re
import json
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MedicalRuleEngine:
    def __init__(self, rules_file_path: str = "data/medical_rules_samples.json"):
        self.rules_file_path = rules_file_path
        self.rewrite_rules = []
        self.rerank_rules = []
        self.normalize_rules = []
        self._load_and_compile_rules()

    def _load_and_compile_rules(self):
        if not os.path.exists(self.rules_file_path):
            logging.error(f"❌ 规则引擎启动失败：找不到外部规则配置文件 {self.rules_file_path}")
            return

        try:
            with open(self.rules_file_path, 'r', encoding='utf-8') as f:
                raw_rules = json.load(f)

            self.rewrite_rules = self._compile_stage_rules(raw_rules.get("rewrite", []))
            self.rerank_rules = self._compile_stage_rules(raw_rules.get("rerank", []))
            self.normalize_rules = self._compile_stage_rules(raw_rules.get("normalize", []))

            total_count = len(self.rewrite_rules) + len(self.rerank_rules) + len(self.normalize_rules)
            logging.info(f"✅ 规则引擎动态加载成功！共编译外部规则 {total_count} 条 "
                         f"(Rewrite: {len(self.rewrite_rules)} | Rerank: {len(self.rerank_rules)} | Normalize: {len(self.normalize_rules)})")

        except Exception as e:
            logging.error(f"❌ 解析外部规则文件时发生严重错误: {e}")
    def _compile_stage_rules(self, rule_list: List[Dict]) -> List[Dict]:
        compiled = []
        for rule in rule_list:
            r_id = rule.get("id", "UNKNOWN")
            pattern = rule.get("pattern", "")
            replacement = rule.get("replacement", "")

            if not pattern:
                continue

            regex_pattern = pattern
            regex_repl = replacement

            # 解析通用泛化占位符（如 [疾病]、[部位]）
            placeholders = re.findall(r"\[(.*?)\]", pattern)
            if placeholders:
                for ph in set(placeholders):
                    clean_ph = ph.strip()
                    regex_pattern = regex_pattern.replace(f"\\[{clean_ph}\\]", f"(?P<{clean_ph}>.+?)")
                    regex_repl = regex_repl.replace(f"[{clean_ph}]", f"\\g<{clean_ph}>")

            try:
                compiled_regex = re.compile(regex_pattern, re.IGNORECASE)
                compiled.append({
                    "id": r_id,
                    "regex": compiled_regex,
                    "replacement": regex_repl
                })
            except Exception as e:
                logging.error(f"⚠️ 规则 [{r_id}] 编译失败: {pattern} -> {e}")

        return compiled

    def rewrite_query(self, raw_text: str) -> Dict[str, Any]:
        processed_text = raw_text.strip()
        triggered_rules = []

        for rule in self.rewrite_rules:
            if rule["regex"].search(processed_text):
                new_text = rule["regex"].sub(rule["replacement"], processed_text)
                if new_text != processed_text:
                    processed_text = new_text
                    triggered_rules.append(rule["id"])

        return {
            "origin_text": raw_text,
            "rewritten_text": processed_text,
            "triggered_rules": list(set(triggered_rules))
        }

    def rerank_candidates(self, rewritten_query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        active_transformations = []
        for rule in self.rerank_rules:
            match = rule["regex"].search(rewritten_query)
            if match:
                expected_target = rule["regex"].sub(rule["replacement"], match.group(0))
                active_transformations.append(expected_target)

        reranked = []
        for item in candidates:
            name = item.get('name', '')
            score = item.get('score', 0.0)
            bonus = 0.0

            for expected in active_transformations:
                if expected in name:
                    bonus += 0.25  # 靶向匹配成功奖励权重
            item['rerank_score'] = score + bonus
            reranked.append(item)

        reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
        return reranked

    def sanitize_output(self, llm_output: str) -> str:
        output = llm_output.strip()

        for rule in self.normalize_rules:
            if rule["regex"].search(output):
                output = rule["regex"].sub(rule["replacement"], output)
        roman_map = {"III": "3", "II": "2", "IV": "4", "I": "1"}
        grade_pattern = re.compile(r"([IVX]+)级")

        def replace_roman(match):
            roman_val = match.group(1)
            return f"{roman_map.get(roman_val, roman_val)}级"
        output = grade_pattern.sub(replace_roman, output)
        return output