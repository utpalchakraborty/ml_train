"""
Pre-tokenize dataset offline. Run on any machine (CPU is fine).
Produces Arrow files that train.py can load directly, skipping tokenization on the GPU machine.
"""

import argparse

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-tokenize dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--dataset_split", type=str, default="train_sft")
    parser.add_argument("--dataset_size", type=int, default=200000)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="tokenized_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading dataset: {args.dataset_name} ({args.dataset_split})")
    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset = dataset.shuffle(seed=args.seed).select(range(args.dataset_size))

    split = dataset.train_test_split(test_size=args.val_split, seed=args.seed)

    def tokenize(example):
        token_ids = tokenizer.apply_chat_template(
            example["messages"],
            truncation=True,
            max_length=args.max_length,
        )
        return {"input_ids": token_ids}

    for name, ds in [("train", split["train"]), ("val", split["test"])]:
        print(f"Tokenizing {name} ({len(ds)} examples)...")
        tokenized = ds.map(
            tokenize,
            remove_columns=ds.column_names,
            num_proc=args.num_proc,
            desc=f"Tokenizing {name}",
        )
        path = f"{args.output_dir}/{name}"
        tokenized.save_to_disk(path)
        print(f"Saved to {path}")

    print("Done. Transfer the output_dir to the GPU machine.")


if __name__ == "__main__":
    main()
