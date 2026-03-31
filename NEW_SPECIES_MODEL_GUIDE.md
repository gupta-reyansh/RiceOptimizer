# 单物种重新训练与新模型产出操作说明

本文档说明如何在当前仓库中，基于已有的 CodonTransformer 预训练模型，对**单一物种**进行重新训练（finetune），并产出可用于后续推理的新模型文件。

下文统一使用：

- `<repo_root>`：CodonTransformer 仓库根目录
- `<work_dir>`：你的单物种训练工作目录，例如 `<repo_root>/workflows/<your_species>`

> 适用范围  
> - 适用于：对仓库当前已支持的物种做单物种继续训练/微调。  
> - 不直接适用于：向当前模型中加入一个**全新、未收录的新物种**。当前代码会校验 `organism` 必须存在于 `CodonTransformer/CodonUtils.py` 的 `ORGANISM2ID` 中，且模型的 `type_vocab_size` 固定为 164 个物种。

---

## 1. 先确认你的目标属于哪一类

### 情况 A：目标物种已经在仓库支持列表中

这类情况可以直接按本文档执行：

1. 准备该物种的 DNA / protein 配对数据。
2. 转成训练所需的 CSV。
3. 调用 `prepare_training_data(...)` 生成 JSON 训练文件。
4. 运行 `finetune.py` 微调模型。
5. 导出最终模型文件并进行推理验证。

### 情况 B：目标物种不在仓库支持列表中

当前仓库**不能只靠准备输入文件就直接完成新物种训练**，原因如下：

1. `prepare_training_data` 会把 `organism` 映射到 `ORGANISM2ID`，未知物种会直接报错。  
2. `finetune.py` 加载的模型基于既有物种 embedding，组织数量由 `NUM_ORGANISMS = 164` 决定。  
3. 推理阶段同样要求物种名称或 ID 已经存在于 `ORGANISM2ID`。

因此，如果你要做“真正的新物种接入”，当前仓库至少还需要：

1. 扩展 `ORGANISM2ID` / `ID2ORGANISM` / `NUM_ORGANISMS`。  
2. 用新的 `type_vocab_size` 构建模型配置。  
3. 重新初始化或迁移物种相关 embedding。  
4. 重新训练并验证新物种是否可正常推理。  

如果你的目标只是“把已有支持物种训练得更贴近你自己的数据”，请继续使用下面的单物种微调流程。

---

## 2. 环境与代码准备

在仓库根目录执行：

```bash
cd <repo_root>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

如果你使用的是集群环境，也可以参考：

- `<repo_root>/slurm/finetune.sh`
- `<repo_root>/slurm/pretrain.sh`

---

## 3. 训练输入文件准备要求

### 3.1 原始输入文件格式

重新训练单一物种时，建议先准备一个 CSV 文件，例如：

`your_species_training.csv`

必须包含以下 3 列：

| 列名 | 含义 | 是否必需 |
|---|---|---|
| `dna` | DNA 序列 | 是 |
| `protein` | 对应蛋白序列 | 是 |
| `organism` | 物种名或物种 ID | 是 |

### 3.2 推荐内容规范

根据仓库当前预处理逻辑，建议遵守以下规则：

1. `dna` 使用大写，仅包含 `A/T/C/G`。  
2. `protein` 使用标准氨基酸单字母大写表示。  
3. `organism` 对单物种训练时，整列应填写同一个物种名称。  
4. `dna` 最好以终止密码子结尾：`TAA` / `TAG` / `TGA`。  
5. `protein` 最好以 `_` 或 `*` 结尾；如果没有，预处理会自动补 `_`。  
6. 一条 `dna` 必须与对应 `protein` 翻译结果一致，否则训练质量会明显下降。  
7. DNA 长度必须是 3 的倍数。  
8. 极短、低质量、含大量模糊碱基的序列应提前清理。  

### 3.3 允许但会被替换的内容

仓库中已有容错逻辑，但建议尽量避免依赖：

1. DNA 中的非 `A/T/C/G` 三联体会被替换成 `UNK`。  
2. 蛋白中的模糊氨基酸会按 `ProteinConfig` 规则做标准化处理。  
3. 若蛋白末尾没有 stop symbol，会自动补 `_`。  
4. 若 DNA 末尾没有 stop codon，预处理阶段会补一个未知 stop token。  

---

## 4. 推荐目录结构

建议先建立一个独立工作目录，便于反复训练与留档：

```text
<repo_root>/workflows/<your_species>/
├── data/
│   ├── raw/
│   │   └── your_species_training.csv
│   ├── processed/
│   │   └── training_data.json
│   └── sample/
│       └── inference_examples.csv
├── checkpoints/
│   ├── finetune.ckpt
│   └── final_model.pt
└── logs/
```

---

## 5. CSV 文件示例

下面是一个最小示例：

```csv
dna,protein,organism
ATGGCTGCTTAA,MAA*,Escherichia coli general
ATGAAATGCTAG,MKC*,Escherichia coli general
```

说明：

- `organism` 必须与 `CodonTransformer/CodonUtils.py` 中的名称完全一致。  
- 如果你训练的是大肠杆菌，优先使用 `Escherichia coli general`。  
- README 中也建议优先选用 `FINE_TUNE_ORGANISMS` 里的物种进行微调。  

---

## 6. 训练前校验清单

在生成训练 JSON 之前，建议先检查：

1. CSV 是否至少包含 `dna`、`protein`、`organism` 三列。  
2. `organism` 是否全部一致。  
3. 物种名称是否能在 `ORGANISM2ID` 中找到。  
4. DNA 长度是否全部为 3 的倍数。  
5. 是否存在空值、重复值、异常短序列。  
6. 是否存在 protein 与 DNA 不匹配的数据。  

可使用如下脚本做基础检查：

```python
import pandas as pd

csv_path = "<work_dir>/data/raw/your_species_training.csv"
df = pd.read_csv(csv_path)

required = {"dna", "protein", "organism"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df[["dna", "protein", "organism"]].isnull().any().any():
    raise ValueError("Null values found; clean the dataset first.")

bad_len = df["dna"].fillna("").map(len) % 3 != 0
if bad_len.any():
    raise ValueError("Some DNA sequences do not have lengths divisible by 3.")

print("记录数:", len(df))
print("物种:", df["organism"].value_counts().to_dict())
```

---

## 7. 生成训练 JSON 文件

CodonTransformer 训练并不直接读取 CSV，而是读取经过预处理后的 JSON 行文件。

执行示例：

```bash
cd <repo_root>
python - <<'PY'
from CodonTransformer.CodonData import prepare_training_data

prepare_training_data(
    "<work_dir>/data/raw/your_species_training.csv",
    "<work_dir>/data/processed/training_data.json",
)
PY
```

生成后的 `training_data.json` 为 JSON Lines 格式，每行类似：

```json
{"idx": 0, "codons": "M_ATG A_GCT A_GCT __TAA", "organism": 51}
```

其中：

- `codons` 是蛋白与密码子对齐后的内部训练表示。  
- `organism` 已经从名称转成整数 ID。  

---

## 8. 运行单物种微调训练

训练入口脚本为：

`<repo_root>/finetune.py`

最小示例：

```bash
cd <repo_root>
python finetune.py \
  --dataset_dir "<work_dir>/data/processed/training_data.json" \
  --checkpoint_dir "<work_dir>/checkpoints" \
  --checkpoint_filename "finetune.ckpt" \
  --batch_size 6 \
  --max_epochs 15 \
  --num_workers 5 \
  --accumulate_grad_batches 1 \
  --num_gpus 4 \
  --learning_rate 0.00005 \
  --warmup_fraction 0.1 \
  --save_every_n_steps 512 \
  --seed 123
```

### 8.1 主要参数说明

| 参数 | 作用 |
|---|---|
| `--dataset_dir` | 训练 JSON 文件路径 |
| `--checkpoint_dir` | checkpoint 输出目录 |
| `--checkpoint_filename` | 保存的 checkpoint 文件名 |
| `--batch_size` | batch size |
| `--max_epochs` | 训练轮数 |
| `--num_workers` | DataLoader worker 数 |
| `--accumulate_grad_batches` | 梯度累积 |
| `--num_gpus` | GPU 数量 |
| `--learning_rate` | 学习率 |
| `--warmup_fraction` | warmup 比例 |
| `--save_every_n_steps` | 每多少 step 存一次 checkpoint |
| `--seed` | 随机种子 |
| `--debug` | 调试模式，会降到单卡并关闭多 worker |

### 8.2 训练产物

训练过程中你通常会得到：

1. `finetune.ckpt`：用于恢复或导出模型。  
2. Lightning 默认输出的日志和中间状态。  

---

## 9. 从 checkpoint 导出最终模型文件

推理时建议导出为 `.pt` 文件，便于后续直接加载。

执行示例：

```bash
cd <repo_root>
python - <<'PY'
from CodonTransformer.CodonPrediction import create_model_from_checkpoint
from CodonTransformer.CodonUtils import NUM_ORGANISMS

create_model_from_checkpoint(
    checkpoint_dir="<work_dir>/checkpoints/finetune.ckpt",
    output_model_dir="<work_dir>/checkpoints/final_model.pt",
    num_organisms=NUM_ORGANISMS,
)
PY
```

导出后你会得到：

- `final_model.pt`

这就是你后续加载的新模型文件。

---

## 10. 使用新模型做推理验证

建议至少用几条已知蛋白序列做 sanity check，确认模型可以正常生成目标物种偏好的 DNA。

```bash
cd <repo_root>
python - <<'PY'
import torch
from transformers import AutoTokenizer
from CodonTransformer.CodonPrediction import predict_dna_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("adibvafa/CodonTransformer")

output = predict_dna_sequence(
    protein="MKTIIALSYIFCLVFADYKDDD*",
    organism="Escherichia coli general",
    device=device,
    tokenizer=tokenizer,
    model="<work_dir>/checkpoints/final_model.pt",
    attention_type="original_full",
    deterministic=True,
)

print(output.organism)
print(output.predicted_dna)
PY
```

### 建议验证的内容

1. 模型是否能正常加载。  
2. 输出 DNA 是否长度正确。  
3. 输出 DNA 是否能翻译回原始蛋白。  
4. 与训练前基线模型相比，目标物种的密码子偏好是否更接近你的数据集。  

---

## 11. 推荐的训练后评估

如果你希望判断“新模型是否真的更适合该物种”，建议至少做下面几项：

1. 用保留的验证集蛋白做推理。  
2. 计算生成序列与目标物种参考密码子频率的一致性。  
3. 对比微调前与微调后的：
   - CAI / CSI
   - GC content
   - Codon Frequency Distribution
4. 检查是否出现异常重复片段或翻译错误。  

仓库中的 `CodonEvaluation` 模块可用于后续评估。

---

## 12. 常见失败原因

### 12.1 `Invalid organism name`

原因：CSV 中的 `organism` 不在 `ORGANISM2ID` 中。  

处理方式：

1. 检查拼写是否与 `CodonTransformer/CodonUtils.py` 完全一致。  
2. 若是全新物种，当前仓库不能直接微调，需要先改造物种映射与模型配置。  

### 12.2 训练文件生成失败

常见原因：

1. CSV 缺列。  
2. 序列为空。  
3. DNA / protein 数据格式不规范。  

### 12.3 训练能跑但效果差

常见原因：

1. 单物种样本量太少。  
2. DNA 与 protein 配对不干净。  
3. 训练集噪声太大。  
4. 目标物种与原始预训练分布差异太大。  

### 12.4 checkpoint 能保存但推理失败

常见原因：

1. 直接把 `.ckpt` 当成标准模型文件使用，但加载方式不匹配。  
2. 没有先导出成 `final_model.pt`。  
3. 推理时使用了不支持的物种名。  

---

## 13. 最终交付物清单

完成一次单物种重新训练后，建议至少保留以下文件：

1. 原始训练 CSV  
2. 预处理后的 `training_data.json`  
3. 训练日志  
4. `finetune.ckpt`  
5. `final_model.pt`  
6. 一份验证样本及预测结果  
7. 一份训练参数记录（batch size、epoch、学习率、seed 等）

---

## 14. 一条完整流程总结

如果你训练的是仓库已支持的物种，完整流程就是：

1. 准备 `dna/protein/organism` 三列 CSV。  
2. 清理并校验序列质量。  
3. 调用 `prepare_training_data(...)` 生成 `training_data.json`。  
4. 运行 `finetune.py` 生成 `finetune.ckpt`。  
5. 调用 `create_model_from_checkpoint(...)` 导出 `final_model.pt`。  
6. 用 `predict_dna_sequence(...)` 做推理验证。  
7. 用评估指标判断新模型是否优于原模型。  

如果你的目标是“接入一个仓库里还不存在的新物种”，那就不是单纯的数据准备问题，而是**需要先扩展物种映射和模型结构**，之后再重新训练。
