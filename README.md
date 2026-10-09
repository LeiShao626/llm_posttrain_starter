# 单卡LLM后训练入门：从手写循环到TRL

这是一个**教学项目**：用本地 12 条中文问答，让同一个 0.5B 模型完成几步 LoRA 监督微调（SFT）。它的目的在于看懂训练代码，不是得到有实用能力的模型。默认一次只处理 1 条样本、最多 128 个 token、训练 12 步；单张V100 32GB 都可以尝试。首次运行需要下载模型。

## 学到什么

| 包 | 功能 | 位置 |
|---|---|---|
| `torch` | `DataLoader`、`AdamW`、梯度与参数更新 | `manual_train.py` 的训练循环 |
| `transformers` | 分词器、聊天模板、预训练因果语言模型 | `manual_train.py` 的加载与预处理 |
| `datasets` | 读取 JSONL，使用 `map` 预处理 | 两个训练脚本的 `load_dataset` |
| `peft` | 用 `LoraConfig` 和 `get_peft_model` 训练 LoRA | `manual_train.py` 的 LoRA 配置 |
| `accelerate` | 设备放置、混合精度、反向传播与启动 | `Accelerator`、`prepare`、`backward` |
| `deepspeed` | 通过 Accelerate 配置 ZeRO-2 | `configs/accelerate_deepspeed.yaml` |
| `trl` | 用 `SFTTrainer` 封装同一个 SFT 任务 | `trl_train.py` |

## 环境

**Linux + NVIDIA GPU**；\
**Python 3.10** 或 **3.11**；\
**V100** 使用 `fp16`。

```bash
conda create -n llm-starter python=3.10 -y
conda activate llm-starter
# 先根据服务器 CUDA 版本安装 PyTorch。
pip install torch==2.5.1 --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu121
pip install -r requirements.txt
```

## 在国内下载模型：选一种方式

训练数据已在 `data/train.jsonl`，无需访问远程数据集。只需下载基座模型。

### 方式 A：HF 镜像

在**启动 Python 进程之前**设置 `HF_ENDPOINT`，随后运行训练脚本。Linux 终端：

```bash
export HF_ENDPOINT=https://hf-mirror.com
accelerate launch --num_processes 1 --mixed_precision fp16 manual_train.py
```

这个设置只作用于当前终端会话。

### 方式 B：魔搭下载

执行：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir ./models/Qwen2.5-0.5B-Instruct
```

之后：

```bash
export STARTER_MODEL=./models/Qwen2.5-0.5B-Instruct
accelerate launch --num_processes 1 --mixed_precision fp16 manual_train.py
python infer.py --adapter outputs/manual
```

## 手写训练循环


```bash
conda create -n tokenizer-lab python=3.10 -y
conda activate tokenizer-lab
pip install "transformers>=4.51,<5" jupyterlab
jupyter lab
```

默认使用 HF 镜像。

```bash
accelerate launch --num_processes 1 --mixed_precision fp16 manual_train.py
python infer.py --adapter outputs/manual --question "什么是梯度下降？"
```

按下面顺序读 `manual_train.py`：

1. `tokenize_example`：聊天模板得到 prompt；`labels=-100` 让 prompt 和 padding 不参与损失。
2. `load_dataset` → `map` → `DataLoader`：文本如何变为一批张量。
3. `from_pretrained` → `get_peft_model`：载入基座，并只让 LoRA 参数可训练。
4. `accelerator.prepare`：让模型、优化器、数据进入同一训练环境。
5. `zero_grad` → `model(**batch).loss` → `backward` → `step`：一个完整的参数更新步。


## 用 TRL 做同一个任务

```bash
python trl_train.py
python infer.py --adapter outputs/trl --question "什么是梯度下降？"
```


## 体验单卡 DeepSpeed

在 **Linux + V100** 上，从项目目录运行：

```bash
accelerate launch --config_file configs/accelerate_deepspeed.yaml manual_train.py --max_steps 12 --output_dir outputs/deepspeed
python infer.py --adapter outputs/deepspeed
```

## 文件与常见问题

- `data/train.jsonl`：12 条本地教学数据；每行都有 `prompt` 和 `completion`。
- `manual_train.py`：重点读，手写更新步骤和标签遮盖。
- `trl_train.py`：对照版；使用相同模型、数据、LoRA 参数和训练步数。
- `infer.py`：加载基座与保存的 LoRA 适配器，生成一条回答。
- `configs/accelerate_deepspeed.yaml`：单卡 DeepSpeed ZeRO-2 配置。
- `outputs/`：运行后自动生成，不需要提交。


## 对照文档

- [Transformers 语言模型训练示例](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/README.md)
- [PEFT LoRA 快速入门](https://huggingface.co/docs/peft/quicktour)
- [Accelerate 的 DeepSpeed 接入](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
- [TRL SFTTrainer 与 prompt/completion 格式](https://huggingface.co/docs/trl/en/sft_trainer)
