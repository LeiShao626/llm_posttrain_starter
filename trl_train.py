"""同一个 SFT 任务的 TRL 写法：对照手写训练循环，观察框架封装了什么。"""

# 标准库负责读取模型路径设置和定位项目目录。
import os
from pathlib import Path

# datasets 读取本地 JSONL；PEFT 配置 LoRA；TRL 管理训练循环。
from datasets import load_dataset
from peft import LoraConfig, TaskType
from trl import SFTConfig, SFTTrainer


# 默认使用 HF 模型 ID，也允许 STARTER_MODEL 指向魔搭下载的本地目录。
MODEL_ID = os.environ.get("STARTER_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ROOT = Path(__file__).resolve().parent


def to_conversation(example: dict) -> dict:
    """把普通问答转成 TRL 的对话式 prompt/completion。

    参数：example 含字符串 prompt 和 completion。
    返回：各含一条 role/content 消息的 prompt、completion。
    """
    # 对话格式让 TRL 自动应用模型的 chat template。
    return {
        "prompt": [{"role": "user", "content": example["prompt"]}],
        "completion": [{"role": "assistant", "content": example["completion"]}],
    }


def main() -> None:
    """创建 SFTTrainer，训练 12 步，并保存 LoRA 适配器。"""
    # datasets 读取与手写版相同的教学数据。
    dataset = load_dataset("json", data_files=str(ROOT / "data" / "train.jsonl"), split="train")
    # map 把原始数据改成 TRL 可识别的对话式问答格式。
    dataset = dataset.map(to_conversation, remove_columns=dataset.column_names)
    # SFTConfig 集中管理训练参数；V100 支持 fp16，但不支持 bf16。
    config = SFTConfig(
        output_dir=str(ROOT / "outputs" / "trl"),
        max_steps=12,
        per_device_train_batch_size=1,
        learning_rate=2e-4,
        max_length=128,
        completion_only_loss=True,
        fp16=True,
        bf16=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    # LoRA 设置与手写版保持一致，便于比较代码接口。
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    # SFTTrainer 自动处理聊天模板、分词、batch、loss 和优化器。
    trainer = SFTTrainer(
        model=MODEL_ID,
        args=config,
        train_dataset=dataset,
        peft_config=lora_config,
    )
    # 触发 TRL 封装好的训练循环。
    trainer.train()
    # 保存的是 LoRA 适配器，不是完整的 0.5B 基座模型。
    trainer.save_model(config.output_dir)


# 直接执行时才开始训练。
if __name__ == "__main__":
    main()
