"""加载手写版保存的 LoRA 适配器，观察一次生成。"""

# 标准库负责解析命令行参数和路径。
import argparse
import os
from pathlib import Path

# torch 负责关闭梯度；PEFT 合并基座和适配器；Transformers 加载模型与分词器。
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# 推理必须加载与训练相同的基座；本地魔搭模型也由 STARTER_MODEL 指定。
MODEL_ID = os.environ.get("STARTER_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ROOT = Path(__file__).resolve().parent


def main() -> None:
    """读取 --adapter 和 --question，打印模型生成的回答。"""
    # 允许选择手写版或 TRL 版输出的 LoRA 目录。
    parser = argparse.ArgumentParser(description="测试 LoRA 适配器")
    parser.add_argument("--adapter", type=Path, default=ROOT / "outputs" / "manual")
    parser.add_argument("--question", default="什么是 LoRA？")
    args = parser.parse_args()
    # 自动选择 GPU 或 CPU；V100 上使用 fp16 减少显存占用。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    # 加载分词器以及训练时保存的特殊 token 设置。
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    # 先加载不含 LoRA 的基座模型。
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)
    # 再把 LoRA 适配器接到基座模型上。
    model = PeftModel.from_pretrained(base_model, args.adapter)
    # 推理模式关闭 dropout 等训练行为。
    model.eval()
    # 使用与训练一致的聊天模板构造输入。
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.question}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)
    # 生成时不需要保存反向传播所用的计算图。
    with torch.no_grad():
        output = model.generate(ids, max_new_tokens=80, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    # 只解码新生成的部分，避免把用户问题也打印出来。
    print(tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True))


# 直接执行文件时才运行推理。
if __name__ == "__main__":
    main()
