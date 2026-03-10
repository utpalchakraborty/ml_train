"""
LoRA SFT training script for Qwen3.5-0.8B-Base on UltraChat.
Designed for RunPod GPU training.
"""

import argparse

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT Training")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--dataset_split", type=str, default="train_sft")
    parser.add_argument("--dataset_size", type=int, default=10000)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "v_proj"])

    # Training config
    parser.add_argument("--output_dir", type=str, default="sft_output")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--save_total_limit", type=int, default=3)

    # W&B
    parser.add_argument("--wandb_project", type=str, default="sft-qwen3.5-0.8b")
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load and prepare dataset
    print(f"Loading dataset: {args.dataset_name} ({args.dataset_split})")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset = dataset.shuffle(seed=args.seed).select(range(args.dataset_size))

    split = dataset.train_test_split(test_size=args.val_split, seed=args.seed)
    train_dataset = split["train"].remove_columns(["prompt", "prompt_id"])
    val_dataset = split["test"].remove_columns(["prompt", "prompt_id"])
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        run_name=args.run_name,
        seed=args.seed,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        args=training_args,
        max_seq_length=args.max_seq_length,
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    final_path = f"{args.output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
