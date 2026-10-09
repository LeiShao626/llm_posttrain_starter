"""手写 SFT：datasets 读数据，Transformers 建模，PEFT 加 LoRA，Accelerate 管设备。"""

# 标准库负责命令行参数和路径；其余导入分别对应项目要学习的五个包。
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


# 模型只有约 0.5B 参数，适合单张 V100 做教学实验。
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# 无论从哪里启动脚本，都以脚本所在目录定位数据。
ROOT = Path(__file__).resolve().parent


def tokenize_example(example: dict, tokenizer, max_length: int) -> dict:
    """把一条问答转成模型输入。

    参数：example 含 prompt/completion；tokenizer 是模型分词器；max_length 是序列上限。
    返回：input_ids、attention_mask、labels；labels 中的 -100 表示该位置不计算 loss。
    """
    # 把用户问题包装成模型熟悉的聊天格式，并留下 assistant 的开头。
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["prompt"]}],
        # 先返回字符串，暂时不转成 token ID
        tokenize=False,
        # 在末尾加上“接下来轮到助手回答”的开头
        add_generation_prompt=True,
        # 最后结果类似这样：
        # <|im_start|>user
        # 什么是 LoRA？<|im_end|>
        # <|im_start|>assistant
    )
    # 只对 assistant 回答计算损失，因此先计算 prompt 占用多少 token。
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    # 把目标回答及结束符接到 prompt 后面，组成完整训练样本。
    full_text = prompt_text + example["completion"] + tokenizer.eos_token
    # 固定长度便于 DataLoader 拼 batch；attention_mask 区分正文和填充。
    # input_ids:    [真实, 真实, 真实, 真实, 真实, 填充, 填充, 填充]
    # attention_mask:  [1, 1, 1, 1, 1, 0, 0, 0]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    # 模型内部会把 labels 错开一位，计算“预测下一个 token”的交叉熵。
    labels = encoded["input_ids"].copy()
    # prompt 和 padding 都不属于目标回答，故用 -100 忽略。
    labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
    labels = [label if mask else -100 for label, mask in zip(labels, encoded["attention_mask"])]
    # 防止 max_length 太短，以至于目标回答被整个截掉。
    if all(label == -100 for label in labels):
        raise ValueError("目标回答被截断；请增大 --max_length")
    # 返回 PyTorch 训练循环所需的三种张量字段。
    return {**encoded, "labels": labels}


def parse_args() -> argparse.Namespace:
    """解析教学实验参数；返回 argparse.Namespace。"""
    # 把常改的训练参数集中放在命令行，避免反复修改源码。
    parser = argparse.ArgumentParser(description="手写 LoRA SFT 训练循环")
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "manual")
    return parser.parse_args()


def main() -> None:
    """加载数据和模型，执行训练循环，然后保存 LoRA 适配器。"""
    # 读取命令行参数。
    args = parse_args()
    # Accelerator 根据启动配置选择 CPU/GPU、混合精度和可选的 DeepSpeed。
    accelerator = Accelerator()
    # 固定随机种子，便于比较两次练习的结果。
    torch.manual_seed(42)
    # Transformers 自动下载模型对应的分词器。
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Qwen 的结束符也用于填充；损失由 labels 的 -100 控制。
    tokenizer.pad_token = tokenizer.eos_token
    # datasets 从本地 JSONL 读取数据，返回可 map 的 Dataset。
    dataset = load_dataset("json", data_files=str(ROOT / "data" / "train.jsonl"), split="train")
    # map 对每条样本执行我们的预处理函数，并去掉原始文本列。
    dataset = dataset.map(
        lambda row: tokenize_example(row, tokenizer, args.max_length),
        remove_columns=dataset.column_names,
    )
    # 让 Dataset 取样时直接返回 torch.Tensor。
    dataset.set_format(type="torch")
    # batch_size=1 让每一步与一条样本对应，方便观察。
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # 模型先以 fp32 加载；Accelerate 在 V100 上用 fp16 自动混合精度运行。
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    # LoRA 只在注意力的 q/v 投影上添加可训练的低秩矩阵。
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    # PEFT 冻结原模型权重，包装出可训练的 LoRA 模型。
    model = get_peft_model(model, lora_config)
    # 打印 LoRA 参数占总参数的比例。
    model.print_trainable_parameters()
    # AdamW 只接收 requires_grad=True 的参数，避免优化冻结权重。
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    # prepare 把模型、优化器和数据装载器放到正确设备，并接入 DeepSpeed（若启用）。
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    # 训练模式会启用 dropout 等训练行为。
    model.train()
    # 教学实验按指定更新步数运行；数据耗尽后从头开始。
    step = 0
    while step < args.max_steps:
        for batch in dataloader:
            # 每个更新步开始前清除旧梯度。
            optimizer.zero_grad()
            # 模型根据 input_ids、attention_mask 和 labels 计算交叉熵损失。
            loss = model(**batch).loss
            # Accelerate 统一处理普通训练和混合精度训练的反向传播。
            accelerator.backward(loss)
            # 优化器根据 LoRA 参数上的梯度更新权重。
            optimizer.step()
            # 只由主进程打印，避免多进程时重复输出。
            accelerator.print(f"step={step + 1:02d} loss={loss.item():.4f}")
            # 完成一个参数更新步。
            step += 1
            # 达到指定步数就结束本轮数据遍历。
            if step >= args.max_steps:
                break
    # 等待所有进程完成后再写文件。
    accelerator.wait_for_everyone()
    # ZeRO-2 下模型参数仍可直接从主进程保存为 PEFT 适配器。
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        accelerator.print(f"LoRA 适配器已保存到 {args.output_dir}")


# 只有直接执行本文件时才开始训练；导入函数不会启动训练。
if __name__ == "__main__":
    main()
