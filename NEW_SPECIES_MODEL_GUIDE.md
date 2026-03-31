# 单物种重新训练与模型产出技术指南

本文档说明如何在当前仓库中，从 CDS FASTA 直接生成训练输入，并对单一物种执行**从头重新训练（retraining / pretraining from scratch）**。文档覆盖环境构建、输入约束、一键脚本使用方式、手工分步流程与常见故障处理。

---

## 1. 适用范围与边界

### 1.1 适用场景

本文档适用于以下场景：

1. 训练目标为针对单一物种从头训练模型，而不是加载 Hugging Face 基础模型继续微调。  
2. 原始输入为目标物种的 CDS FASTA，或已经整理完成的基因级别 CDS 序列集合。  
3. 目标物种可以不在 `/home/runner/work/CodonTransformer/CodonTransformer/CodonTransformer/CodonUtils.py` 的全局 `ORGANISM2ID` 中。  

### 1.2 不适用场景

本文档不直接覆盖**将新物种永久接入仓库默认推理接口**的场景。当前仓库的默认推理流程仍依赖固定物种映射：

1. 公开基础模型与默认推理入口围绕既有 `ORGANISM2ID` 设计。  
2. 若需要把新物种永久加入公共多物种模型与默认推理接口，仍需同步修改物种映射与相关配置。  

因此，本文档中的一键脚本解决的是**单物种从头训练**问题，而不是永久扩展仓库的公共多物种模型定义。

---

## 2. 当前仓库中的训练逻辑概述

单物种重新训练的正式输入并不是 FASTA，而是两级中间产物：

1. **训练 CSV**：包含 `dna`、`protein`、`organism` 三列。  
2. **训练 JSONL**：由 `CodonTransformer.CodonData.prepare_training_data(...)` 生成，供 `pretrain.py` 读取。  

当前的一键脚本执行以下流程：

1. `read_fasta_file(...)` 使用 Biopython 读取 FASTA，并按指定或推断的 codon table 翻译出蛋白序列。  
2. 过滤无法正确翻译或长度不合法的记录（除非显式要求保留）。  
3. 生成训练 CSV。  
4. 调用 `prepare_training_data(...)` 生成训练 JSONL，并将当前物种映射为本次训练的局部 organism id `0`。  
5. 调用 `pretrain.py` 从头初始化 BigBird 掩码语言模型并开始训练。  

与此前的单物种微调流程不同，当前一键脚本默认直接使用过滤后的全部合格 CDS 记录进行重新训练，不再执行面向微调场景的 CSI Top 10% 子集筛选。

---

## 3. 环境构建

### 3.1 基础软件要求

- Python `>=3.9`
- `pip`
- 可用的 CUDA / GPU 运行环境（`pretrain.py` 当前固定使用 GPU 加速）
- 若不提供本地 tokenizer 文件，则需要具备访问 Hugging Face tokenizer 仓库的网络环境

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

若 `torch.cuda.is_available()` 返回 `False`，当前 `pretrain.py` 无法在默认配置下启动训练。

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

输入物种名称会被写入训练数据和运行元数据。对于一键重新训练流程，它**不必**预先存在于全局 `ORGANISM2ID` 中；脚本会自动去除首尾空白，并将该物种映射为本次训练的局部 organism id `0`。

例如：

- `Fragaria vesca`
- `Escherichia coli general`
- `Homo sapiens`

若后续需要把训练出的模型接入仓库默认推理接口，仍需再处理全局物种映射问题。

### 4.4 密码子表要求

默认情况下，`read_fasta_file(...)` 会根据物种名推断翻译用的 NCBI codon table。对新的或未显式列出的物种，建议通过一键脚本的 `--codon_table` 参数显式指定密码子表，避免错误翻译。例如植物核基因常用 table `1`，多数细菌常用 table `11`。

### 4.5 关于表达量、FPKM、TPM 等附加信息

当前仓库的 `prepare_training_data(...)` 仅使用以下三列：

- `dna`
- `protein`
- `organism`

因此，FPKM、TPM、组织来源、处理条件等字段不会直接进入模型输入。此类信息可以用于样本筛选，但不会由当前训练脚本直接建模。

---

## 5. 一键重新训练脚本

仓库根目录脚本：

`/home/runner/work/CodonTransformer/CodonTransformer/train_species_model.py`

该脚本完成以下流程：

1. 读取 CDS FASTA。  
2. 按指定或推断的密码子表翻译蛋白序列。  
3. 默认过滤 `correct_seq=False` 的记录。  
4. 生成训练 CSV。  
5. 生成训练 JSONL。  
6. 调用 `pretrain.py` 启动从头训练。  
7. 在工作目录中保存中间文件、训练元数据与检查点。  

### 5.1 最小命令

```bash
cd /home/runner/work/CodonTransformer/CodonTransformer

python train_species_model.py \
  --input_fasta "/absolute/path/to/species_cds.fasta" \
  --organism "Fragaria vesca" \
  --codon_table 1
```

在该用法下，脚本会自动创建默认工作目录：

```text
/home/runner/work/CodonTransformer/CodonTransformer/workflows/fragaria_vesca/
```

### 5.2 常用命令

```bash
cd /home/runner/work/CodonTransformer/CodonTransformer

python train_species_model.py \
  --input_fasta "/absolute/path/to/species_cds.fasta" \
  --organism "Fragaria vesca" \
  --work_dir "/absolute/path/to/workflows/fragaria_vesca_run01" \
  --codon_table 1 \
  --batch_size 6 \
  --max_epochs 5 \
  --num_workers 5 \
  --num_gpus 1 \
  --learning_rate 5e-5 \
  --warmup_fraction 0.1 \
  --save_interval 5 \
  --seed 123
```

### 5.3 若需保留 `correct_seq=False` 的记录

```bash
python train_species_model.py \
  --input_fasta "/absolute/path/to/species_cds.fasta" \
  --organism "Fragaria vesca" \
  --codon_table 1 \
  --keep_all_records
```

> 建议仅在明确理解数据质量风险的前提下启用 `--keep_all_records`。

### 5.4 脚本参数说明

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--input_fasta` | CDS FASTA 文件路径 | 必填 |
| `--organism` | 目标物种名称，会映射为本次训练的局部 organism id `0` | 必填 |
| `--work_dir` | 工作目录 | `<repo_root>/workflows/<organism_slug>` |
| `--tokenizer_path` | 本地 tokenizer.json 路径；为空时加载默认 Hugging Face tokenizer | 空 |
| `--codon_table` | 显式指定 FASTA 翻译使用的 NCBI codon table | 自动推断 |
| `--keep_all_records` | 是否保留 `correct_seq=False` 记录 | 关闭 |
| `--batch_size` | 训练 batch size | `6` |
| `--max_epochs` | 训练轮数 | `5` |
| `--num_workers` | DataLoader worker 数 | `5` |
| `--accumulate_grad_batches` | 梯度累积步数 | `1` |
| `--num_gpus` | 使用 GPU 数量 | `1` |
| `--learning_rate` | 学习率 | `5e-5` |
| `--warmup_fraction` | warmup 比例 | `0.1` |
| `--save_interval` | 按 epoch 保存检查点的间隔 | `5` |
| `--seed` | 随机种子 | `123` |
| `--debug` | 透传给 `pretrain.py` 的调试模式 | 关闭 |

---

## 6. 一键脚本的输出结构

默认情况下，工作目录会生成如下内容：

```text
<work_dir>/
├── checkpoints/
│   └── epoch_*.ckpt
├── data/
│   ├── processed/
│   │   └── training_data.json
│   └── raw/
│       ├── parsed_cds_records.csv
│       └── training_sequences.csv
├── logs/
└── run_metadata.json
```

各文件含义如下：

- `parsed_cds_records.csv`：FASTA 解析后的原始记录，包括翻译结果与 `correct_seq` 标记。  
- `training_sequences.csv`：最终用于重新训练的三列表格（`dna/protein/organism`）。  
- `training_data.json`：供 `pretrain.py` 读取的 JSONL 数据。  
- `run_metadata.json`：本次运行的参数、物种名、局部 organism id 与样本规模摘要。  
- `checkpoints/`：模型重新训练输出目录。  

---

## 7. 手工分步流程

若不使用一键脚本，也可以手工执行：

1. 用 `read_fasta_file(...)` 从 FASTA 提取 `dna/protein/organism`。  
2. 过滤不合格记录并导出训练 CSV。  
3. 调用 `prepare_training_data(..., organism_to_id={target_species: 0})` 生成训练 JSONL。  
4. 调用 `pretrain.py --train_data_path ... --type_vocab_size 1` 启动训练。  

示例：

```python
from CodonTransformer.CodonData import prepare_training_data, read_fasta_file

species = "Fragaria vesca"
df = read_fasta_file(
    input_file="/absolute/path/to/species_cds.fasta",
    organism=species,
    codon_table=1,
)
df = df[df["correct_seq"]].copy()
df = df[["dna", "protein"]].copy()
df["organism"] = species
prepare_training_data(
    dataset=df,
    output_file="/absolute/path/to/training_data.json",
    organism_to_id={species: 0},
)
```

随后运行：

```bash
python pretrain.py \
  --train_data_path "/absolute/path/to/training_data.json" \
  --checkpoint_dir "/absolute/path/to/checkpoints" \
  --type_vocab_size 1
```

---

## 8. 常见故障处理

### 8.1 `--organism` 前后有空格

脚本会自动对 `--organism` 执行 `strip()`，例如：

```bash
--organism "Fragaria vesca "
```

会被规范化为：

```text
Fragaria vesca
```

### 8.2 新物种翻译结果异常

若目标物种未被内置的 codon table 推断规则准确覆盖，应显式指定：

```bash
--codon_table 1
```

或其它正确的 NCBI codon table 编号。

### 8.3 `Invalid organism name` 或全局物种映射错误

一键重新训练流程不再要求目标物种提前存在于全局 `ORGANISM2ID`。若仍看到这类错误，通常说明运行的不是更新后的 `train_species_model.py`，或者调用的是默认的多物种推理/旧微调流程。

### 8.4 训练前测试失败

仓库现有测试依赖 `pandas`、`torch`、`ipywidgets` 等包。若测试在导入阶段失败，应先执行：

```bash
pip install -r /home/runner/work/CodonTransformer/CodonTransformer/requirements.txt
```

### 8.5 无 GPU 可用

当前 `pretrain.py` 使用 GPU trainer 配置。若环境无 GPU，需要先调整底层训练脚本，再进行 CPU 训练。

---

## 9. 结论

当前仓库中的 `train_species_model.py` 现已面向**单物种重新训练**而非**单物种微调**。对于 `Fragaria vesca` 这类不在全局 `ORGANISM2ID` 中的新物种，可直接使用 CDS FASTA 触发训练，并通过 `--codon_table` 指定正确翻译规则。若后续需要把该新物种永久纳入仓库默认推理接口，仍需额外扩展全局物种映射与相关推理配置。
