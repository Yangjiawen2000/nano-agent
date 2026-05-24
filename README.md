# Causal Reasoning Agent for Brain-Targeted Nanoparticle Design

**基于因果推理的智能 Agent 用于脑靶向纳米粒子理性设计优化**

---

## 项目简介

本项目构建一个因果推理驱动的 Agent 系统，用于优化脑靶向纳米粒子（NP）的设计参数，以最大化 BBB（血脑屏障）穿透效率。

核心思路：将优化问题从"黑箱搜索"转变为"因果推理驱动的假设-验证循环"。

---

## 项目结构

```
├── pbpk_simulator.py        # Phase 2: PBPK 仿真器（受体介导转胞吞 ODE 模型）
├── causal_graph.py          # Phase 3: 因果图学习（机制驱动 + OLS 权重估计）
├── agent.py                 # Phase 4: ReAct Agent（DeepSeek / Anthropic 双后端）
├── data/
│   ├── np_brain_dataset_final.csv   # 107条文献实验数据
│   ├── causal_results.json          # 学习到的因果图（JSON）
│   ├── agent_trace.json             # Agent 优化轨迹（最新运行结果）
│   ├── causal_dag_final.png         # 因果 DAG 可视化
│   ├── causal_corr_heatmap.png      # 相关性热图
│   ├── pbpk_sensitivity.png         # PBPK 参数敏感性分析图
│   ├── build_starter_dataset.py     # 数据集构建脚本
│   └── 数据集来源文档.md             # 数据溯源文档
└── 07_研究方案/
    └── 研究方向介绍.md               # 研究方向说明文档
```

---

## 系统架构

```
文献实验数据 (107条)
       ↓ Phase 1
  np_brain_dataset_final.csv
       ↓ Phase 2
  pbpk_simulator.py          ← 核心仿真引擎
  · 5状态ODE（受体介导转胞吞）
  · Spearman ρ = 0.912 验证
       ↓ Phase 3
  causal_graph.py            ← 因果图学习
  · 2000条 PBPK 合成数据
  · 机制驱动 DAG + OLS 权重
  · 10/10 条边生物学验证通过
       ↓ Phase 4 ✅
  agent.py                   ← ReAct 推理-行动循环
  · DeepSeek / Anthropic 双后端
  · 5 工具：pbpk_simulate, query_causal_graph, lookup_parameter,
           check_feasibility, compare_designs
  · 演示：62x AUC_brain 提升（0.028→1.727）
```

---

## 快速开始

```bash
pip install numpy scipy matplotlib networkx pandas causal-learn scikit-learn openai
```

### 运行 PBPK 仿真

```python
from pbpk_simulator import pbpk_simulate

result = pbpk_simulate(
    size_nm=80, zeta_mv=-15, peg='yes',
    ligand_type='transferrin', ligand_density=30
)
print(f"AUC_brain = {result['AUC']:.4f}")
print(f"AUC_ratio = {result['AUC_ratio']:.4f}")
```

### 运行因果图学习

```python
python causal_graph.py
# 输出: data/causal_dag_final.png, data/causal_results.json
```

### 运行 ReAct Agent（Phase 4）

```bash
export DEEPSEEK_API_KEY=sk-your-key-here   # 或 ANTHROPIC_API_KEY
python agent.py
```

```python
from agent import run_agent

result = run_agent(
    initial_design={
        "size_nm": 150, "zeta_mv": 5, "peg": 0,
        "ligand_type": "transferrin", "ligand_density": 120,
    },
    goal="Maximise AUC_brain for BBB-targeted NP",
    max_iterations=10,
)
print(f"Best AUC  = {result['best_AUC']:.4f}")
print(f"Best design = {result['best_design']}")
# Best AUC  = 1.7272
# Best design = {size_nm:80, zeta_mv:-15, peg:'yes', ligand_type:'transferrin', ligand_density:30}
```

### 调用 CausalMemory

```python
from causal_graph import load_causal_memory
import numpy as np

memory = load_causal_memory()
print(memory.query_bottleneck('AUCbrain'))

recs = memory.recommend_intervention({
    'LogSize': np.log(80), 'Zeta': -15.0,
    'PEG': 1.0, 'LogLigDensity': np.log1p(30)
})
for r in recs:
    print(f"{r['direction'].upper()} {r['parameter']}: effect={r['total_effect']:+.4f}")
```

---

## PBPK 模型

**状态变量**（5个，均归一化）：

| 变量 | 含义 |
|------|------|
| NP_blood | 血液循环中的 NP（初值=1.0）|
| R_free | BBB 游离受体比例（初值=1.0）|
| NP_R | 受体结合态 NP |
| NP_vesicle | 内体囊泡中的 NP |
| NP_brain | 脑实质累积 NP（目标）|

**动力学参数**（7个）：k_bind, k_off, k_endo, k_trans, k_lyso, k_recycle, CL

---

## 因果 DAG

学习到的 9 节点因果图（设计参数 → PBPK 中介 → 脑内递送）：

```
LogSize ──────────► Kbind ───────────────► AUCbrain
LogLigDensity ────► Kbind
Zeta ─────────────► Ktrans ──────────────► AUCbrain
Zeta ─────────────► Klyso ───────────────► AUCbrain
LogSize ──────────► CL ─────────────────► AUCbrain
PEG ──────────────► CL
```

---

## 数据来源

| 来源 | 数据量 | 指标 |
|------|--------|------|
| Wiley PNAS 2013 | 8条 | 脑内相对计数 |
| Wiley PNAS 2015 | 8条 | BBB 过膜% |
| Khung ACS Nano 2023 | 5条 | BBB 过膜% |
| Nájera Pharmaceutics 2025 | 12条 | %ID/g |
| Yousfan Mol. Pharm. 2024 | 74条 | AUC 比值 / μg·h/mL |

---

## 引用

> Yang J. (2026). A Causal Reasoning Agent for Interpretable Optimization of Brain-Targeted Nanoparticle Design. *(in preparation)*

---

## 许可证

MIT License
