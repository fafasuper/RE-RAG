
import os
import logging
import time
from typing import List

try:
    from openai import OpenAI
except ImportError:
    logging.warning("未安装 openai 库，请执行: pip install openai")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MedicalLLM:
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.0):
        self.api_key = os.environ.get("OPENAI_API_KEY", "your_api_key_here")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model_name = model_name
        self.temperature = temperature

        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            logging.info(f"🚀 初始化 LLM 模块成功 | 模型: {self.model_name}")
        except Exception as e:
            logging.error(f"❌ 初始化 LLM 客户端失败: {e}")

    def _build_prompt(self, raw_text: str, candidates: List[str], is_baseline: bool) -> tuple:
        role_setting = "你是一个医学术语标准化专家。请将输入的临床诊断文本标准化为ICD-10标准术语。"
        few_shot_examples = """示例1：输入：偏执狂
输出：妄想性障碍

示例2：输入：卵巢Ca
输出：卵巢恶性肿瘤

示例3：输入：内耳耳石症
输出：良性阵发性位置性眩晕
"""

        rules = """【处理规则】：
1. 如果【临床诊断】包含多个独立的疾病（如复合诊断），请将其拆分，并用双井号 "##" 连接（例如：急性阑尾炎##弥漫性腹膜炎）。
2. 只输出最终的标准术语名称，绝对不要包含任何多余的解释、标点、前缀或思考过程。"""

        if not is_baseline and candidates:
            candidates_str = "\n".join([f"- {cand}" for cand in candidates])
            rules += "\n3. 请参考下方提供的【候选标准术语集】，优先从中选择最精准匹配的术语；若无完全匹配项，再基于你的医学知识给出标准ICD名称。"
            context_section = f"【候选标准术语集】：\n{candidates_str}\n"
        else:
            context_section = "【候选标准术语集】：无外部参考，请调动你的内部医学知识。\n"

        sys_prompt = f"{role_setting}\n{rules}\n\n【Few-shot 示例】:\n{few_shot_examples}"
        user_prompt = f"{context_section}\n【输入临床诊断】：{raw_text}\n输出："

        return sys_prompt, user_prompt

    def generate(self, raw_text: str, candidates: List[str] = None, is_baseline: bool = False,
                 max_retries: int = 3) -> str:
        if not raw_text:
            return ""

        if candidates is None:
            candidates = []

        sys_prompt, user_prompt = self._build_prompt(raw_text, candidates, is_baseline)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=64,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                logging.warning(f"⚠️ LLM API 调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logging.error(f"❌ LLM 调用彻底失败。")
                    return candidates[0] if candidates and not is_baseline else raw_text