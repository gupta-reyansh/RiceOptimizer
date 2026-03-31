# 单物种微调与模型产出技术指南

本文档说明如何在当前仓库中，基于既有 CodonTransformer 预训练模型完成**单物种微调**，并从 CDS FASTA 直接生成训练输入、训练数据集及最终模型检查点。文档同时覆盖环境构建、输入约束、自动化脚本使用方式、人工分步流程与常见故障处理。

---

## 1. 适用范围与边界

### 1.1 适用场景

本文档适用于以下场景：

1. 目标物种已存在于 `/home/runner/work/CodonTransformer/CodonTransformer/CodonTransformer/CodonUtils.py` 的 `ORGANISM2ID` 中。  
2. 训练目标为在既有基础模型上执行单物种微调，而非重新进行多物种预训练。  
3. 原始输入为目标物种的 CDS FASTA，或已经整理完成的基因级别 CDS 序列集合。  

### 1.2 不适用场景

本文档**不直接适用于真正的新物种接入**。当前仓库的训练与推理流程均依赖固定物种映射：

1. `prepare_training_data(...)` 会将 `organism` 映射为 `ORGANISM2ID` 中的整数 ID。  
2. `finetune.py` 加载的模型使用固定的 `type_vocab_size=164`。  
3. 推理阶段同样要求物种名称已收录在既有映射表中。  

因此，若目标物种未包含在 `ORGANISM2ID` 中，仅通过准备输入文件无法完成训练，还需要同步修改物种映射、模型配置与物种 embedding。

---

## 2. 当前仓库中的训练逻辑概述

单物种微调的正式输入并不是 FASTA，而是两级中间产物：

1. **训练 CSV**：必须仅包含 `dna`、`protein`、`organism` 三列。  
2. **训练 JSONL**：由 `CodonTransformer.CodonData.prepare_training_data(...)` 生成，供 `finetune.py` 读取。  

当前仓库的数据处理逻辑如下：

1. `read_fasta_file(...)` 使用 Biopython 读取 FASTA，并基于物种对应的密码子表翻译出蛋白序列。  
2. `prepare_training_data(...)` 将 `dna` 与 `protein` 合并为模型使用的 codon token 序列，并将 `organism` 转换为固定 ID。  
3. `finetune.py` 自动从 Hugging Face 下载基础模型和 tokenizer，并执行单物种微调。  

根据论文及仓库说明，单物种微调更推荐使用目标物种中 **CSI（Codon Similarity Index）最高的前 10% 基因**，而不是直接使用全部基因。因此，本文档提供的一键脚本默认执行 CSI 评分并保留前 10% 候选序列。

---

## 3. 环境构建

### 3.1 基础软件要求

- Python `>=3.9`
- `pip`
- 可用的 CUDA / GPU 运行环境（`finetune.py` 当前固定使用 GPU 加速）
- 具备访问 Hugging Face 模型仓库的网络环境，或已提前缓存基础模型

### 3.2 推荐构建方式

在仓库根目录 `/home/runner/work/CodonTransformer/CodonTransformer` 执行：

```bash
cd /home/runner/work/CodonTransformer/CodonTransformer

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

### 3.3 环境验证

建议执行以下命令验证关键依赖是否可用：

```bash
python - <<'PY'
import Bio
import pandas
import torch
import pytorch_lightning
import transformers

print("Biopython:", Bio.__version__)
print("pandas:", pandas.__version__)
print("torch:", torch.__version__)
print("Environment check passed.")
PY
```

### 3.4 GPU 验证

建议进一步确认 PyTorch 可以识别 GPU：

```bash
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
PY
```

若 `torch.cuda.is_available()` 返回 `False`，当前 `finetune.py` 无法在默认配置下启动训练。

---

## 4. 输入数据要求

### 4.1 输入文件类型

一键训练脚本接收**CDS FASTA** 文件作为原始输入。每条记录应代表一条基因级 CDS，而不是整条染色体、contig 或 scaffold。

### 4.2 序列质量要求

建议输入数据满足以下条件：

1. DNA 序列使用大写字母，仅包含 `A/T/C/G`。  
2. 序列长度应为 3 的倍数。  
3. 数据应尽量以合法起始密码子开始，并以终止密码子结束。  
4. 数据集中应尽量避免内部异常 stop codon、低质量注释和明显错误拼接。  
5. 若 FASTA 中包含大量不完整或无法正确翻译的序列，建议在训练前先做质量清洗。  

### 4.3 物种名称要求

输入物种名称必须与 `ORGANISM2ID` 中的名称**完全一致**。例如：

- `Escherichia coli general`
- `Homo sapiens`
- `Saccharomyces cerevisiae`

若物种名无法匹配，训练流程会在预处理阶段直接终止。

### 4.4 关于表达量、FPKM、TPM 等附加信息

当前仓库的 `prepare_training_data(...)` 仅使用以下三列：

- `dna`
- `protein`
- `organism`

因此，FPKM、TPM、组织来源、处理条件等字段不会直接进入模型输入。此类信息可以用于样本筛选、排序或构建高表达子集，但不会由当前训练脚本直接建模。

---

## 5. 一键训练脚本

仓库根目录新增脚本：

`/home/runner/work/CodonTransformer/CodonTransformer/train_species_model.py`

该脚本完成以下流程：

1. 读取 CDS FASTA。  
2. 按指定物种的密码子表翻译蛋白序列。  
3. 默认过滤 `correct_seq=False` 的记录。  
4. 计算 CSI，并默认保留前 10% 高 CSI 样本。  
5. 生成训练 CSV。  
6. 生成训练 JSONL。  
7. 调用 `finetune.py` 启动模型微调。  
8. 在工作目录中保存中间文件、训练元数据与检查点。  

### 5.1 最小命令

```bash
cd /home/runner/work/CodonTransformer/CodonTransformer

python train_species_model.py \
  --input_fasta "/absolute/path/to/species_cds.fasta" \
  --organism "Escherichia coli general"
```

在该用法下，脚本会自动创建默认工作目录：

```text
/home/runner/work/CodonTransformer/CodonTransformer/workflows/escherichia_coli_general/
```

### 5.2 常用命令

```bash
cd /home/runner/work/CodonTransformer/CodonTransformer

python train_species_model.py \
  --input_fasta "/absolute/path/to/species_cds.fasta" \
  --organism "Escherichia coli general" \
  --work_dir "/absolute/path/to/workflows/ecoli_general_run01" \
  --top_fraction 0.1 \
  --batch_size 6 \
  --max_epochs 15 \
  --num_workers 5 \
  --num_gpus 1 \
  --learning_rate 5e-5 \
  --warmup_fraction 0.1 \
  --save_every_n_steps 512 \
  --seed 123
```

### 5.3 若需使用全部合格序列

```bash
python train_species_model.py \
  --input_fasta "/absolute/path/to/species_cds.fasta" \
  --organism "Escherichia coli general" \
  --top_fraction 1.0
```

### 5.4 若需保留 `correct_seq=False` 的记录

```bash
python train_species_model.py \
  --input_fasta "/absolute/path/to/species_cds.fasta" \
  --organism "Escherichia coli general" \
  --keep_all_records
```

> 建议仅在明确理解数据质量风险的前提下启用 `--keep_all_records`。

### 5.5 脚本参数说明

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--input_fasta` | CDS FASTA 文件路径 | 必填 |
| `--organism` | 目标物种名称，必须存在于 `ORGANISM2ID` | 必填 |
| `--work_dir` | 工作目录 | `<repo_root>/workflows/<organism_slug>` |
| `--top_fraction` | 保留的高 CSI 样本比例 | `0.1` |
| `--keep_all_records` | 是否保留 `correct_seq=False` 记录 | 关闭 |
| `--checkpoint_filename` | 微调检查点文件名 | `finetune.ckpt` |
| `--batch_size` | 训练 batch size | `6` |
| `--max_epochs` | 训练轮数 | `15` |
| `--num_workers` | DataLoader worker 数 | `5` |
| `--accumulate_grad_batches` | 梯度累积步数 | `1` |
| `--num_gpus` | 使用 GPU 数量 | `1` |
| `--learning_rate` | 学习率 | `5e-5` |
| `--warmup_fraction` | warmup 比例 | `0.1` |
| `--save_every_n_steps` | 检查点保存间隔 | `512` |
| `--seed` | 随机种子 | `123` |
| `--debug` | 透传给 `finetune.py` 的调试模式 | 关闭 |

---

## 6. 一键脚本的输出结构

默认情况下，工作目录会生成如下内容：

```text
<work_dir>/
├── checkpoints/
│   └── finetune.ckpt
├── data/
│   ├── processed/
│   │   └── training_data.json
│   └── raw/
│       ├── parsed_cds_records.csv
│       ├── scored_cds_records.csv
│       └── training_sequences.csv
├── logs/
└── run_metadata.json
```

各文件含义如下：

- `parsed_cds_records.csv`：FASTA 解析后的原始记录，包括翻译结果与 `correct_seq` 标记。  
- `scored_cds_records.csv`：经过过滤并附带 CSI 分数的候选记录。  
- `training_sequences.csv`：最终用于训练的三列 CSV。  
- `training_data.json`：`prepare_training_data(...)` 生成的 JSONL 训练文件。  
- `finetune.ckpt`：微调阶段保存的模型检查点。  
- `run_metadata.json`：训练输入、样本数量、输出路径等结构化元数据。  

---

## 7. 手工分步流程

在需要分步排查、插入额外过滤逻辑或与外部流程集成时，可采用手工流程。

### 7.1 将 CDS FASTA 转为训练 CSV

仓库中已有脚本：

`/home/runner/work/CodonTransformer/CodonTransformer/fasta_to_training_csv.py`

示例：

```bash
cd /home/runner/work/CodonTransformer/CodonTransformer

python fasta_to_training_csv.py \
  --input_fasta "/absolute/path/to/species_cds.fasta" \
  --organism "Escherichia coli general" \
  --output_csv "/absolute/path/to/training_sequences.csv"
```

该脚本默认仅导出 `correct_seq=True` 的记录，并生成项目所需的三列：

- `dna`
- `protein`
- `organism`

### 7.2 从 CSV 生成训练 JSONL

```python
from CodonTransformer.CodonData import prepare_training_data

prepare_training_data(
    "/absolute/path/to/training_sequences.csv",
    "/absolute/path/to/training_data.json",
)
```

### 7.3 启动微调

```bash
cd /home/runner/work/CodonTransformer/CodonTransformer

python finetune.py \
  --dataset_dir "/absolute/path/to/training_data.json" \
  --checkpoint_dir "/absolute/path/to/checkpoints" \
  --checkpoint_filename "finetune.ckpt" \
  --batch_size 6 \
  --max_epochs 15 \
  --num_workers 5 \
  --accumulate_grad_batches 1 \
  --num_gpus 1 \
  --learning_rate 5e-5 \
  --warmup_fraction 0.1 \
  --save_every_n_steps 512 \
  --seed 123
```

---

## 8. 训练策略建议

### 8.1 样本选择建议

推荐优先使用以下样本：

1. 可正确翻译的 CDS。  
2. 长度完整、无明显注释异常的基因。  
3. CSI 较高、能够代表目标物种优选密码子使用模式的基因。  

### 8.2 关于 CSI 过滤

默认保留前 10% 的高 CSI 样本，原因如下：

1. 与论文中的微调策略保持一致。  
2. 可优先学习更接近天然高表达基因的密码子分布。  
3. 在样本质量参差不齐时，通常比“全量无差别纳入”更稳健。  

若目标是最大化样本覆盖，或目标物种本身只有很小规模的数据集，可将 `--top_fraction` 调整为 `1.0`。

### 8.3 关于物种选择

尽管所有 `ORGANISM2ID` 中的物种都可通过当前流程完成 ID 映射，但仓库中同时定义了 `FINE_TUNE_ORGANISMS`。若目标物种位于该列表中，通常更符合原项目公开的微调使用场景。

---

## 9. 常见故障与处理建议

### 9.1 报错：`Unsupported organism`

原因：输入物种名称不在 `ORGANISM2ID` 中。  

处理建议：

1. 检查拼写、空格、大小写和株系描述是否完全一致。  
2. 优先使用 `CodonTransformer/CodonUtils.py` 中已经定义的标准名称。  
3. 若目标物种未收录，则需要修改代码与模型配置，不能直接调用现有微调流程。  

### 9.2 报错：`No eligible CDS records remained after filtering`

原因：

1. FASTA 记录全部被判定为 `correct_seq=False`。  
2. DNA 长度不满足 3 的倍数。  
3. 输入文件并非基因级 CDS，而是基因组片段或低质量序列集合。  

处理建议：

1. 先查看 `parsed_cds_records.csv`。  
2. 检查 `correct_seq` 列及翻译结果。  
3. 仅在明确需要时使用 `--keep_all_records`。  

### 9.3 报错：PyTorch / CUDA / GPU 不可用

原因：`finetune.py` 当前固定使用 GPU 加速。  

处理建议：

1. 确认 CUDA 驱动与 PyTorch 版本匹配。  
2. 确认运行环境中存在可见 GPU。  
3. 在集群环境中，确保作业脚本已经正确申请 GPU 资源。  

### 9.4 首次运行下载模型失败

原因：`finetune.py` 会从 Hugging Face 下载基础模型和 tokenizer。  

处理建议：

1. 确认运行节点具备外网访问权限。  
2. 预先缓存 Hugging Face 模型文件。  
3. 在受限网络环境中，优先采用镜像、离线缓存或预下载策略。  

---

## 10. 结论

当前仓库已经具备“支持物种的单物种微调”所需的核心能力，但其输入要求实际上是 `dna/protein/organism` 三列训练 CSV，而不是原始 FASTA。新增的一键脚本将 CDS FASTA 解析、CSI 筛选、训练数据构建与模型微调串联为单一入口，可用于从原始 CDS 文件直接产出目标物种的微调模型检查点。

对于**已收录物种**，推荐优先采用本文档的一键流程。对于**未收录物种**，则需要先完成物种映射、模型配置与 embedding 扩展，再设计新的训练方案。
