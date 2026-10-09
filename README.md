# 单卡大模型后训练入门：从手写循环到 TRL

这是一个**教学项目**：用本地 12 条中文问答，让同一个 0.5B 模型完成几步 LoRA 监督微调（SFT）。它的目的在于看懂训练代码，不是得到有实用能力的模型。默认一次只处理 1 条样本、最多 128 个 token、训练 12 步；单张 V100 32GB 足够练习。首次运行需要联网下载模型。

## 你会学到什么

| 包 | 在本项目中的最小职责 | 看哪里 |
|---|---|---|
| `torch` | `DataLoader`、`AdamW`、梯度与参数更新 | `manual_train.py` 的训练循环 |
| `transformers` | 分词器、聊天模板、预训练因果语言模型 | `manual_train.py` 的加载与预处理 |
| `datasets` | 读取 JSONL，使用 `map` 预处理 | 两个训练脚本的 `load_dataset` |
| `peft` | 用 `LoraConfig` 和 `get_peft_model` 训练 LoRA | `manual_train.py` 的 LoRA 配置 |
| `accelerate` | 设备放置、混合精度、反向传播与启动 | `Accelerator`、`prepare`、`backward` |
| `deepspeed` | 通过 Accelerate 配置 ZeRO-2 | `configs/accelerate_deepspeed.yaml` |
| `trl` | 用 `SFTTrainer` 封装同一个 SFT 任务 | `trl_train.py` |

## 环境

建议在 **Linux + NVIDIA GPU** 上运行，Python 3.10 或 3.11。V100 使用 `fp16`，不要改为 `bf16`。Windows 原生环境可读代码与尝试普通训练；DeepSpeed 练习建议在 Linux 服务器上做。

```bash
conda create -n llm-starter python=3.10 -y
conda activate llm-starter
# 先根据服务器 CUDA 版本安装 PyTorch；下面仅示范 CUDA 12.1。
pip install torch==2.5.1 --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu121
pip install -r requirements.txt
```

如果服务器 CUDA 版本不同，先按 [PyTorch 安装页](https://pytorch.org/get-started/locally/) 选择对应命令。进入本目录后执行下述命令。模型下载自 [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)。

## 第一关：手写训练循环

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

**练习**：暂时遮住训练循环，自己把上面第 5 步写出来；随后打印 `batch["input_ids"].shape`、`batch["labels"]` 和可训练参数比例。看到训练 loss 下降只说明程序能拟合这几条数据，不能证明泛化能力。

## 第二关：用 TRL 做同一个任务

```bash
python trl_train.py
python infer.py --adapter outputs/trl --question "什么是梯度下降？"
```

对照两个脚本，找出 `SFTTrainer` 替你完成的分词、拼 batch、优化器创建、反向传播和保存。TRL 版使用对话式 `prompt`/`completion` 数据，`completion_only_loss=True` 表示只对回答部分计损失。两种实现处理聊天模板与 token 边界的细节可能略有差别，所以**不要要求两条 loss 曲线逐点相同**。

## 第三关：体验单卡 DeepSpeed

在 **Linux + V100** 上，从项目目录运行：

```bash
accelerate launch --config_file configs/accelerate_deepspeed.yaml manual_train.py --max_steps 12 --output_dir outputs/deepspeed
python infer.py --adapter outputs/deepspeed
```

同一个 `manual_train.py` 不需要改动：Accelerate 根据 YAML 接入 DeepSpeed ZeRO-2。**单卡 ZeRO-2 主要是学习接口，通常没有明显的显存收益，还可能更慢**；对这个 0.5B 模型不需要靠它才能训练。若 DeepSpeed 安装受服务器 CUDA 编译环境影响，先完成前两关。

## 文件与常见问题

- `data/train.jsonl`：12 条本地教学数据；每行都有 `prompt` 和 `completion`。
- `manual_train.py`：重点读，手写更新步骤和标签遮盖。
- `trl_train.py`：对照版；使用相同模型、数据、LoRA 参数和训练步数。
- `infer.py`：加载基座与保存的 LoRA 适配器，生成一条回答。
- `configs/accelerate_deepspeed.yaml`：单卡 DeepSpeed ZeRO-2 配置。
- `outputs/`：运行后自动生成，不需要提交。

如果显存不足，优先把 `--max_length` 从 128 改成 64；如果回答被完全截断，代码会报错，此时需增大长度。若模型下载失败，检查服务器能否访问 Hugging Face，或提前将模型下载到本地并修改两个训练脚本及 `infer.py` 的 `MODEL_ID`。

下一步再回到 dOPSD：先读它的普通 SFT 基线，再研究“生成 → 遮盖 → 教师和学生各跑一次前向 → 计算蒸馏损失”相对本项目新增了什么。

## 对照文档

- [Transformers 语言模型训练示例](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/README.md)
- [PEFT LoRA 快速入门](https://huggingface.co/docs/peft/quicktour)
- [Accelerate 的 DeepSpeed 接入](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
- [TRL SFTTrainer 与 prompt/completion 格式](https://huggingface.co/docs/trl/en/sft_trainer)
