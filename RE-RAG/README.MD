# RE-RAG：中文医学术语标准化与规则增强检索框架

本项目提出了一种基于“多级知识过滤与推理增强（RE-RAG）”架构的中文临床诊断术语标准化方法。通过高度自动化的流水线与消融实验设计，验证规则引擎、混合检索与大模型生成的协同效能。

---

## 📂 项目架构

核心逻辑由 YAML 配置文件动态驱动，实验结果按时间戳自动隔离归档。

```text
med_norm_rag/
├─ configs/                  # 实验消融配置文件 (YAML)
│  ├─ step1_baseline.yaml    # 基线: 纯大模型 Few-shot
│  ├─ step2_rag.yaml         # 基础 RAG: 混合检索 + 大模型
│  └─ step3_re_rag.yaml      # RE-RAG: 规则改写 + 混合检索 + 规则校验 + 大模型
├─ data/                     # 真实数据集与知识库 
│  ├─ chip-cdn/              # 数据集
├─ rules/                    # 规则配置文件目录
│  └─ medical_rules.csv      # 专家规则库
├─ rules_engine/             # 规则引擎模块
│  └─ engine.py              # 前置改写 (Rewrite) 与后置校验 (Normalize)
├─ retrieval/                # 混合检索模块
│  └─ hybrid_search.py       # 稀疏 + 稠密 双路召回
├─ models/                   # 大模型交互模块
│  └─ llm_generator.py       # OpenAI / 本地模型 API 封装与 Prompt 构建
├─ evaluation/               # 评测分析模块
│  └─ evaluator.py           # 多维核心指标计算
├─ main_pipeline.py          # 统一主流水线入口
└─ requirements.txt          # 环境依赖

```

## 环境与数据准备
CHIP-CDN和Yiduyun数据需要自行申请下载

CHIP-CDN数据集：https://tianchi.aliyun.com/cblue；
医渡云标准化7K数据集：http://data.openkg.cn/dataset/yidu-n7k

## 运行
### 安装依赖
`pip install -r requirements.txt`

### 准备规则库
pip data/build_knowledge_base.py
pip data/filter_semantic_leaps.py

准备好icd_lib.json  medical_rules.csv等文件

###  根据不同实验步骤选择不同配置文件
示例：python main_pipeline.py --config configs/step1_baseline.yaml

## 评估指标说明

### 主指标（CHIP-CDN）：
Strict Accuracy：预测集合 == gold 集合
Sample-wise F1：逐样本 set-F1 后求平均（macro over samples）
Micro-F1：对全体标签汇总的 micro（辅助指标）

### 检索/排序诊断指标：

Hit@k：gold 集合是否被 top-k 候选完全覆盖（严格上限）
P@k：top-k 候选是否命中 gold 任一标签（宽松召回）
Oracle sample-F1：候选集合与 gold 的交集能达到的上限（用于判断瓶颈在召回还是选择）









