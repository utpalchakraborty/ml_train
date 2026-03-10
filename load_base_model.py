from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, PeftModel
from transformers import TrainingArguments

base_model_name = "Qwen/Qwen3.5-0.8B-Base"
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto").to("mps")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)



messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"},
      {"role": "assistant", "content": "The capital of France is Paris."},
      {"role": "user", "content": "What about Germany?"},
  ]

token_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("mps")

outputs = model.generate(**token_ids, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))