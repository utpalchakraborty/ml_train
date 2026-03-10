.PHONY: train eval setup

setup:
	uv sync
	wandb login

train:
	uv run python train.py \
		--dataset_size 200000 \
		--lora_r 64 \
		--lora_alpha 128 \
		--lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
		--per_device_train_batch_size 4 \
		--gradient_accumulation_steps 16 \
		--max_length 1024 \
		--num_epochs 3

eval:
	uv run python eval.py --lora_path sft_output/final
