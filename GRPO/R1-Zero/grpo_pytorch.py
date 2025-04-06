import torch
import torch.nn as nn
import torch.optim as optim
import re
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import random
import wandb
import os
from typing import List, Dict, Any, Tuple, Callable, Optional

# Constants
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

class GSM8KDataset(Dataset):
    def __init__(self, split="train", tokenizer=None):
        self.tokenizer = tokenizer
        self.data = self.get_gsm8k_questions(split)
    
    def extract_hash_answer(self, text: str) -> str | None:
        if "####" not in text:
            return None
        return text.split("####")[1].strip().replace(",", "").replace("$", "")
    
    def get_gsm8k_questions(self, split = "train"):
        data = load_dataset('openai/gsm8k', 'main')[split]
        data = data.map(lambda x: {
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': x['question']}
            ],
            'answer': self.extract_hash_answer(x['answer'])
        })
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'prompt': item['prompt'],
            'answer': item['answer']
        }

class RewardFunctions:
    @staticmethod
    def extract_xml_answer(text: str) -> str:
        if "<answer>" not in text or "</answer>" not in text:
            return ""
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    
    @staticmethod
    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [RewardFunctions.extract_xml_answer(r) for r in responses]
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

    @staticmethod
    def int_reward_func(completions, **kwargs) -> list[float]:
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [RewardFunctions.extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

    @staticmethod
    def strict_format_reward_func(completions, **kwargs) -> list[float]:
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
        return [0.5 if match else 0.0 for match in matches]

    @staticmethod
    def soft_format_reward_func(completions, **kwargs) -> list[float]:
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
        return [0.5 if match else 0.0 for match in matches]

    @staticmethod
    def count_xml(text) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
            count -= len(text.split("\n</answer>\n")[-1])*0.001
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
        return count

    @staticmethod
    def xmlcount_reward_func(completions, **kwargs) -> list[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [RewardFunctions.count_xml(c) for c in contents]

class GRPOTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        reward_funcs,
        train_dataset,
        output_dir="outputs/pytorch-grpo",
        run_name="pytorch-grpo",
        learning_rate=5e-6,
        weight_decay=0.1,
        adam_beta1=0.9,
        adam_beta2=0.99,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=16,
        max_prompt_length=256,
        max_completion_length=786,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=100,
        max_grad_norm=0.1,
        device="cuda",
        use_wandb=False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.train_dataset = train_dataset
        self.output_dir = output_dir
        self.run_name = run_name
        
        # Training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.warmup_ratio = warmup_ratio
        self.lr_scheduler_type = lr_scheduler_type
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_generations = num_generations
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.num_train_epochs = num_train_epochs
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.use_wandb = use_wandb
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.adam_beta1, self.adam_beta2),
            weight_decay=self.weight_decay
        )
        
        # DataLoader setup
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.per_device_train_batch_size,
            shuffle=True
        )
        
        # Setup learning rate scheduler
        total_steps = len(self.train_dataloader) * self.num_train_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        self.scheduler = self._get_lr_scheduler(total_steps, warmup_steps)
        
        if self.use_wandb:
            wandb.init(project=self.run_name)
    
    def _get_lr_scheduler(self, total_steps, warmup_steps):
        if self.lr_scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, total_steps)
        else:
            # Default to linear
            return optim.lr_scheduler.LinearLR(
                self.optimizer, 
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=warmup_steps
            )
    
    def _generate_samples(self, prompt_batch, answer_batch):
        """Generate multiple samples for each prompt using multinomial sampling"""
        generations = []
        
        for prompt, answer in zip(prompt_batch, answer_batch):
            # Prepare prompt
            chat_prompt = self.tokenizer.apply_chat_template(
                prompt, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate multiple responses for this prompt
            with torch.no_grad():
                outputs = self.model.generate(
                    chat_prompt,
                    max_new_tokens=self.max_completion_length,
                    do_sample=True,
                    num_return_sequences=self.num_generations,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            
            # Process outputs
            batch_generations = []
            for output in outputs:
                # Extract only the generated text (without prompt)
                gen_text = self.tokenizer.decode(
                    output[len(chat_prompt[0]):], 
                    skip_special_tokens=True
                )
                
                # Format like the trl API expects
                batch_generations.append([{'content': gen_text}])
            
            generations.append({
                'prompt': prompt,
                'answer': answer,
                'generations': batch_generations
            })
        
        return generations
    
    def _compute_rewards(self, generation_data):
        """Compute rewards using multiple reward functions"""
        all_rewards = []
        
        for data in generation_data:
            prompt = data['prompt']
            answer = data['answer']
            completions = data['generations']
            
            # Compute rewards from each reward function
            rewards = []
            for reward_func in self.reward_funcs:
                reward = reward_func(prompts=[prompt]*len(completions), 
                                    completions=completions,
                                    answer=[answer]*len(completions))
                rewards.append(reward)
            
            # Transpose to get list of rewards per completion
            rewards_per_completion = list(zip(*rewards))
            
            # Sum rewards for each completion
            total_rewards = [sum(r) for r in rewards_per_completion]
            all_rewards.append(total_rewards)
        
        return all_rewards
    
    def _compute_grpo_loss(self, generation_data, rewards):
        """Compute GRPO loss based on rewards"""
        total_loss = 0
        
        for data, batch_rewards in zip(generation_data, rewards):
            prompt = data['prompt']
            completions = data['generations']
            
            # Convert to tensor and normalize rewards
            rewards_tensor = torch.tensor(batch_rewards, device=self.device)
            normalized_rewards = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
            
            for completion, reward in zip(completions, normalized_rewards):
                # Format input for the model
                chat_input = prompt + completion
                model_input = self.tokenizer.apply_chat_template(
                    chat_input, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Get completion token positions
                prompt_len = len(self.tokenizer.apply_chat_template(prompt, return_tensors="pt")[0])
                completion_mask = torch.zeros_like(model_input)
                completion_mask[:, prompt_len:] = 1
                
                # Forward pass
                outputs = self.model(model_input, labels=model_input)
                logits = outputs.logits
                
                # Standard language model loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = model_input[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                               shift_labels.view(-1))
                
                # Apply mask to only consider completion tokens
                shift_mask = completion_mask[..., 1:].contiguous().view(-1)
                masked_loss = loss * shift_mask
                
                # Scale loss by reward
                scaled_loss = masked_loss.sum() * -reward
                
                total_loss += scaled_loss / (shift_mask.sum() + 1e-8)
        
        return total_loss / len(generation_data)
    
    def train(self):
        """Run the GRPO training loop"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        global_step = 0
        self.model.train()
        
        for epoch in range(self.num_train_epochs):
            epoch_loss = 0
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
            
            for step, batch in enumerate(progress_bar):
                # Generate samples
                generation_data = self._generate_samples(
                    batch['prompt'], 
                    batch['answer']
                )
                
                # Compute rewards
                rewards = self._compute_rewards(generation_data)
                
                # Compute loss
                loss = self._compute_grpo_loss(generation_data, rewards)
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                epoch_loss += loss.item()
                
                # Update model parameters
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Logging
                    if global_step % self.logging_steps == 0:
                        avg_loss = epoch_loss / (step + 1)
                        progress_bar.set_postfix({"loss": avg_loss})
                        
                        if self.use_wandb:
                            wandb.log({
                                "loss": avg_loss,
                                "learning_rate": self.scheduler.get_last_lr()[0],
                                "epoch": epoch
                            })
                    
                    # Save model checkpoint
                    if global_step % self.save_steps == 0:
                        checkpoint_dir = os.path.join(
                            self.output_dir, 
                            f"checkpoint-{global_step}"
                        )
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        self.model.save_pretrained(checkpoint_dir)
                        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save final model
        final_dir = os.path.join(self.output_dir, "final_model")
        os.makedirs(final_dir, exist_ok=True)
        self.model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        
        return global_step, epoch_loss / len(self.train_dataloader)

# Main execution
def main():
    # Model configuration
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # or "meta-llama/Llama-3.2-1B-Instruct"
    output_dir = "outputs/pytorch-grpo-qwen"
    run_name = "pytorch-grpo-qwen-gsm8k"
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # Create dataset
    dataset = GSM8KDataset(tokenizer=tokenizer)
    
    # Initialize reward functions
    reward_funcs = [
        RewardFunctions.xmlcount_reward_func,
        RewardFunctions.soft_format_reward_func,
        RewardFunctions.strict_format_reward_func,
        RewardFunctions.int_reward_func,
        RewardFunctions.correctness_reward_func
    ]
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=reward_funcs,
        train_dataset=dataset,
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        per_device_train_batch_size=4,  # Lower batch size for direct PyTorch implementation
        gradient_accumulation_steps=16, # Increased to compensate for smaller batch size
        num_generations=16,
        max_prompt_length=256,
        max_completion_length=786,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=100,
        max_grad_norm=0.1,
        use_wandb=False  # Set to True to enable wandb logging
    )
    
    # Train the model
    trainer.train()

if __name__ == "__main__":
    main() 