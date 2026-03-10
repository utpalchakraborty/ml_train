"""
Evaluate a LoRA-finetuned model vs base model side-by-side.
"""

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


EVAL_PROMPTS = [
    # Simple single-turn
    [{"role": "user", "content": "What is the capital of France?"}],
    # Instruction following
    [{"role": "user", "content": "Explain photosynthesis in 2-3 sentences."}],
    # Multi-turn
    [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What about Germany?"},
    ],
    # System prompt + instruction
    [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "List 3 benefits of exercise."},
    ],
]


def generate(model, tokenizer, messages, device, max_new_tokens=200):
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(device)
    outputs = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    # Decode only the new tokens
    new_tokens = outputs[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA model")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype="auto"
    ).to(args.device)

    # Load LoRA model
    print(f"Loading LoRA adapter from {args.lora_path}...")
    lora_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype="auto"
    )
    lora_model = PeftModel.from_pretrained(lora_model, args.lora_path).to(args.device)

    print("=" * 80)
    for i, messages in enumerate(EVAL_PROMPTS):
        print(f"\n--- Prompt {i+1} ---")
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        print(f"User: {user_msgs[-1]}")

        base_out = generate(base_model, tokenizer, messages, args.device, args.max_new_tokens)
        lora_out = generate(lora_model, tokenizer, messages, args.device, args.max_new_tokens)

        print(f"\n[BASE]: {base_out}")
        print(f"\n[LORA]: {lora_out}")
        print("=" * 80)


if __name__ == "__main__":
    main()
