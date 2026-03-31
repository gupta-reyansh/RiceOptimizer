# 单物种重新训练与新模型产出操作说明

本文档说明如何在当前仓库中，基于已有的 CodonTransformer 预训练模型，对**单一物种**进行重新训练（finetune），并产出可用于后续推理的新模型文件。

本文档已结合仓库内文章 `/home/runner/work/CodonTransformer/CodonTransformer/article.md` 的训练思路补充整理。论文中的 CodonTransformer 基础模型使用约 100 万条 DNA-protein 配对序列、164 个物种进行训练；官方微调示例则重点使用目标物种中 **CSI（Codon Similarity Index）最高的前 10% 基因**，以得到更贴近高表达天然基因分布的模型。

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
4. `dna` 最好以合法起始密码子开头，例如 `ATG`；如果你希望尽量贴近论文的数据标准，建议只保留以起始密码子开始的 CDS。  
5. `dna` 最好以终止密码子结尾：`TAA` / `TAG` / `TGA`。  
6. `dna` 中间不要出现额外 stop codon；论文中的训练数据在预处理中保留的是**长度可被 3 整除、以 start codon 开头、以 stop codon 结尾、且仅包含一个 stop codon** 的序列。  
7. `protein` 最好以 `_` 或 `*` 结尾；如果没有，预处理会自动补 `_`。  
8. 一条 `dna` 必须与对应 `protein` 翻译结果一致，否则训练质量会明显下降。  
9. DNA 长度必须是 3 的倍数。  
10. 极短、低质量、含大量模糊碱基的序列应提前清理。  
11. 单条序列不要超过模型最大长度。论文与代码都基于最长 2048 token 的输入；对应到蛋白-密码子对后，超长序列会被截断，不适合作为高质量训练样本。  

### 3.3 允许但会被替换的内容

仓库中已有容错逻辑，但建议尽量避免依赖：

1. DNA 中的非 `A/T/C/G` 三联体会被替换成 `UNK`。  
2. 蛋白中的模糊氨基酸会按 `ProteinConfig` 规则做标准化处理。  
3. 若蛋白末尾没有 stop symbol，会自动补 `_`。  
4. 若 DNA 末尾没有 stop codon，预处理阶段会补一个未知 stop token。  

### 3.4 按论文思路筛选更适合微调的数据

如果你的目标是尽量复现论文中的微调思路，建议不要把目标物种的全部基因都直接拿来训练，而是优先筛选**CSI 较高的基因子集**。

推荐做法：

1. 先为目标物种所有候选基因计算 CSI。  
2. 按 CSI 从高到低排序。  
3. 优先选择前 10% 作为微调集，或至少构建一个“高 CSI 子集”版本与“全量数据”版本做对照。  

这样做的原因是：

1. 论文中的官方微调就是基于高 CSI 基因进行的。  
2. 高 CSI 基因更接近目标物种的优选密码子使用习惯。  
3. 微调后的模型更容易生成接近天然高表达基因分布的序列，并在部分场景下降低负向 cis-elements。  

如果你暂时没有 CSI 计算流程，也建议至少先做一个高质量子集：

- 去掉明显低表达、低质量、注释不完整或翻译不一致的样本  
- 优先保留可信 CDS  
- 先用高质量小集合训练，再和全量集合训练做对比

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
5. DNA 是否以合法 start codon 开头。  
6. DNA 是否只在末尾保留一个 stop codon。  
7. 是否存在空值、重复值、异常短序列。  
8. 是否存在 protein 与 DNA 不匹配的数据。  
9. 是否有超长序列会触发截断。  
10. 如果要贴近论文方案，是否已经完成 CSI 排序，并保留了高 CSI 子集。  

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

如果你已经有目标物种的 codon usage table，可进一步增加：

1. 计算每条基因的 CSI  
2. 导出 CSI 前 10% 的样本  
3. 同时保留全量版与高 CSI 版，后续分别微调并比较结果

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

上面的参数与论文描述的官方微调设置基本一致：

- batch size = 6  
- learning rate 峰值 = `5e-5`  
- warmup 比例 = `0.1`  
- 训练轮数 = `15`  
- 使用 4 张 GPU 进行微调  

如果你是首次训练某个物种，建议先按论文设置跑出一个基线版本，再逐步调整。

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

### 8.3 如何理解微调后的结果变化

结合论文结果，微调并不一定表现为“CSI 越高越好”，而更像是把模型输出**拉向你提供的训练子集分布**。

这意味着：

1. 如果你用目标物种的高 CSI 基因做微调，模型通常会更接近该高质量子集的密码子分布。  
2. 某些物种上，微调后 CSI 可能提升；另一些物种上，微调后更可能是向天然高表达基因模式靠拢，而不是单纯无限提高 CSI。  
3. 因此评估时不能只看一个分数，最好同时看 CSI、GC、CFD、局部 codon pattern 和负向 cis-elements。  

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
4. 评估局部密码子分布是否更像天然基因。  
5. 检查负向 cis-elements 是否增加。  
6. 检查是否出现异常重复片段或翻译错误。  

仓库中的 `CodonEvaluation` 模块可用于后续评估。

### 11.1 论文中重点关注的评估指标

除了常见的 CAI / CSI / GC 外，论文还特别强调以下指标：

#### 1. CSI（Codon Similarity Index）

- 用于衡量序列与目标物种整体密码子使用频率表的相似度  
- 论文的微调集选择核心就是“目标物种 CSI 最高的前 10% 基因”  
- 适合用于筛选训练样本，也适合用于比较 base model 与 finetune model

#### 2. CFD（Codon Frequency Distribution）

- 用于衡量稀有密码子的比例  
- 论文里把低于最常见同义密码子 30% 使用频率的密码子视为 rare codon  
- 如果你的训练目标是更贴近天然表达模式，而不是一味追求高频密码子，CFD 很重要

#### 3. %MinMax profile

- 这是论文里用来观察**局部 codon 分布模式**的核心指标  
- 它不是只看“哪些 codon 被用到了”，而是看序列沿线低频/高频密码子的分布  
- 论文使用长度为 18 codons 的滑动窗口

#### 4. DTW（Dynamic Time Warping）

- 用来比较生成序列与天然序列的 `%MinMax` 曲线距离  
- 如果微调后模型的 normalized DTW 更低，通常表示生成序列的局部密码子分布更接近天然基因

#### 5. Negative cis-elements

- 论文强调异源表达时应尽量减少 negative cis-regulatory elements  
- 这是 CodonTransformer 相比部分商业工具的重要优势之一  
- 可使用 GenScript 的 rare codon / sequence analysis 工具做辅助检查

#### 6. RNA minimum free energy（可选高级评估）

- 论文还比较了生成序列对应 RNA 的最小自由能  
- 如果你有条件做更深入评估，可以把微调前后模型生成的 RNA folding energy 与天然序列做对比

### 11.2 一个更贴近论文的评估顺序

建议按下面顺序评估：

1. **翻译正确性**：生成 DNA 是否能翻译回原蛋白  
2. **全局使用偏好**：CSI / CAI / GC 是否落在合理范围  
3. **稀有密码子比例**：CFD 是否异常升高或过低  
4. **局部分布模式**：%MinMax 与 DTW 是否更接近天然高质量基因  
5. **调控风险**：negative cis-elements 是否增加  
6. **高级结构指标**：RNA minimum free energy 是否偏离天然序列太多

### 11.3 数据准备时为什么不能只追求“高频密码子越多越好”

论文的核心结论之一是：真正好的优化结果不仅要匹配目标物种的整体 codon preference，还要尽量保留**天然的局部 codon pattern**。

这意味着：

1. 不能只靠“每个氨基酸都选最高频 codon”来构建训练集或评估标准。  
2. 稀有 codon 并不一定是坏事，局部高低频搭配会影响翻译节奏、折叠和表达稳定性。  
3. 因此训练数据最好来自天然高质量基因，而不是人工极端优化后的序列。  

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
