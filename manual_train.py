"""手写 SFT：datasets 读数据，Transformers 建模，PEFT 加 LoRA，Accelerate 管设备。"""

# 标准库负责命令行参数和路径；其余导入分别对应项目要学习的五个包。
import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


# 默认从 Hugging Face 下载；STARTER_MODEL 也可指向魔搭下载的本地目录。
MODEL_ID = os.environ.get("STARTER_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
# 无论从哪里启动脚本，都以脚本所在目录定位数据。
ROOT = Path(__file__).resolve().parent


def tokenize_example(example: dict, tokenizer, max_length: int) -> dict:
    """把一条问答转成模型输入。

    参数：example 含 prompt/completion；tokenizer 是模型分词器；max_length 是序列上限。
    返回：input_ids、attention_mask、labels；labels 中的 -100 表示该位置不计算 loss。

    贯穿例子：example = {"prompt": "什么是 LoRA？", "completion": "LoRA 是低秩微调。"}。
    下方的 P1/A1/PAD 是为了看清流程而写的“示意 token”，不是 Qwen 真实的 token ID。
    假设 prompt 被分成 [P1,P2,P3,P4]，回答与结束符被分成 [A1,A2,A3,EOS]，
    且 max_length=12，那么最后得到：
      input_ids:      [P1,P2,P3,P4,A1,A2,A3,EOS,PAD,PAD,PAD,PAD]
      attention_mask: [ 1, 1, 1, 1, 1, 1, 1,  1,  0,  0,  0,  0]
      labels:         [-100,-100,-100,-100,A1,A2,A3,EOS,-100,-100,-100,-100]
    真正运行时 token 数量和整数 ID 由 Qwen 分词器决定，可能与示意不同。
    """
    # 第 1 步：从 example["prompt"] 取出「什么是 LoRA？」并包成一条 user 消息。
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["prompt"]}],
        # tokenize=False：此时要的是可读字符串，还不是 token ID。
        tokenize=False,
        # add_generation_prompt=True：在末尾加 assistant 的开头，准备接回答。
        add_generation_prompt=True,
    )
    # 第 1 步结束：prompt_text 大致是
    #   <|im_start|>user\n什么是 LoRA？<|im_end|>\n<|im_start|>assistant\n
    # 第 2 步：把 prompt_text 转成整数 ID；模板已有特殊标记，因此不额外添加。
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    # 第 2 步结束：prompt_ids 示意为 [P1,P2,P3,P4]，len(prompt_ids)=4。
    # 第 3 步：接上目标回答和结束符；模型将学习生成这一段。
    full_text = prompt_text + example["completion"] + tokenizer.eos_token
    # 第 3 步结束：full_text 大致是 prompt_text +「LoRA 是低秩微调。」+ <|im_end|>。
    # 第 4 步：把完整文本转成 ID；太长则截断，太短则补到 max_length。
    encoded = tokenizer(
        full_text,
        # full_text 已含聊天模板标记与 eos 标记，不再自动追加。
        add_special_tokens=False,
        # 超过 max_length 时，截掉末尾多出的 token。
        truncation=True,
        # 本例为方便说明假设 max_length=12；正常训练默认是 128。
        max_length=max_length,
        # 不足 max_length 时，在右侧补 PAD，便于组成 batch。
        padding="max_length",
    )
    # 第 4 步结束：encoded["input_ids"] 示意为 [P1,P2,P3,P4,A1,A2,A3,EOS,PAD,PAD,PAD,PAD]。
    # 同时 encoded["attention_mask"] 为 [1,1,1,1,1,1,1,1,0,0,0,0]。
    # 第 5 步：先复制完整 ID，作为下一 token 预测任务的目标标签。
    labels = encoded["input_ids"].copy()
    # 第 5 步结束：labels 暂时与 input_ids 相同；模型内部会自动错位计算下一个 token 的损失。
    # 第 6 步：前 len(prompt_ids) 个位置属于提问和 assistant 开头；用 -100 忽略它们。
    labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
    # 第 6 步结束：labels 示意为 [-100,-100,-100,-100,A1,A2,A3,EOS,PAD,PAD,PAD,PAD]。
    # 第 7 步：attention_mask 为 0 的位置是补齐区；也改成 -100，不让 PAD 参与 loss。
    labels = [label if mask else -100 for label, mask in zip(labels, encoded["attention_mask"])]
    # 第 7 步结束：labels 示意为 [-100,-100,-100,-100,A1,A2,A3,EOS,-100,-100,-100,-100]。
    # 第 8 步：如果全是 -100，说明目标回答被截光；这种样本无法训练。
    if all(label == -100 for label in labels):
        # 明确提示调大长度，而不是让模型产生难定位的异常。
        raise ValueError("目标回答被截断；请增大 --max_length")
    # 第 9 步：把分词结果和 labels 一起交给 Dataset；稍后才会变成 torch.Tensor。
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
        # 任务是因果语言建模：根据前面的 token 预测下一个 token
        task_type=TaskType.CAUSAL_LM,
        # 秩
        r=8,
        # 控制 LoRA 更新的缩放。这里默认缩放比例是 alpha / r = 16 / 8 = 2
        lora_alpha=16,
        # 训练时对进入 LoRA 分支的输入做 5% 的 dropout
        lora_dropout=0.05,
        # 只在注意力模块中名字为 q_proj（查询投影）和 v_proj（值投影）的层加入 LoRA
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
    # 只让主训练进程执行保存。单卡运行时，它通常就是当前进程；多卡时可以避免多个进程同时写同一目录
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        accelerator.print(f"LoRA 适配器已保存到 {args.output_dir}")


# 只有直接执行本文件时才开始训练；导入函数不会启动训练。
if __name__ == "__main__":
    main()
