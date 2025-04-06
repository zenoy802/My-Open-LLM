# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# os.environ['HF_HOME'] = '/root/autodl-tmp/.cache'

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = DPOConfig(
    output_dir="Qwen2.5-1.5B-DPO",
    logging_steps=10,
    gradient_accumulation_steps=4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()