import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset
import numpy as np
from grpo_pytorch import GRPOTrainer as PyTorchGRPOTrainer, GSM8KDataset, RewardFunctions

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

def prepare_test_cases():
    test_data = load_dataset('openai/gsm8k', 'main')['test'].select(range(10))
    test_cases = []
    
    for item in test_data:
        prompt = [
            {'role': 'system', 'content': """
            Respond in the following format:
            <reasoning>
            ...
            </reasoning>
            <answer>
            ...
            </answer>
            """},
            {'role': 'user', 'content': item['question']}
        ]
        answer = extract_hash_answer(item['answer'])
        test_cases.append({
            'prompt': prompt,
            'answer': answer
        })
    
    return test_cases

def evaluate_trl_model(model, tokenizer, test_cases):
    """Evaluate the TRL implementation on test cases."""
    results = []
    
    for case in test_cases:
        prompt = case['prompt']
        gold_answer = case['answer']
        
        # Format prompt with tokenizer
        inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to("cuda")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=786,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        # Extract answer
        extracted_answer = RewardFunctions.extract_xml_answer(response)
        
        # Calculate reward
        correctness = 1.0 if extracted_answer == gold_answer else 0.0
        
        results.append({
            'response': response,
            'extracted_answer': extracted_answer,
            'gold_answer': gold_answer,
            'correctness': correctness
        })
    
    return results

def evaluate_pytorch_model(model, tokenizer, test_cases):
    """Evaluate the PyTorch implementation on test cases."""
    results = []
    
    for case in test_cases:
        prompt = case['prompt']
        gold_answer = case['answer']
        
        # Format prompt with tokenizer
        inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to("cuda")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=786,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        # Extract answer
        extracted_answer = RewardFunctions.extract_xml_answer(response)
        
        # Calculate reward
        correctness = 1.0 if extracted_answer == gold_answer else 0.0
        
        results.append({
            'response': response,
            'extracted_answer': extracted_answer,
            'gold_answer': gold_answer,
            'correctness': correctness
        })
    
    return results

def main():
    # Load test cases
    test_cases = prepare_test_cases()
    
    # Model configuration
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Load TRL trained model
    trl_model_path = "outputs/Qwen-1.5B-GRPO/final_model"
    trl_model = AutoModelForCausalLM.from_pretrained(
        trl_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    trl_tokenizer = AutoTokenizer.from_pretrained(trl_model_path)
    trl_tokenizer.pad_token = trl_tokenizer.eos_token
    
    # Load PyTorch trained model
    pytorch_model_path = "outputs/pytorch-grpo-qwen/final_model"
    pytorch_model = AutoModelForCausalLM.from_pretrained(
        pytorch_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    pytorch_tokenizer = AutoTokenizer.from_pretrained(pytorch_model_path)
    pytorch_tokenizer.pad_token = pytorch_tokenizer.eos_token
    
    # Evaluate both models
    trl_results = evaluate_trl_model(trl_model, trl_tokenizer, test_cases)
    pytorch_results = evaluate_pytorch_model(pytorch_model, pytorch_tokenizer, test_cases)
    
    # Compare results
    trl_accuracy = sum(r['correctness'] for r in trl_results) / len(trl_results)
    pytorch_accuracy = sum(r['correctness'] for r in pytorch_results) / len(pytorch_results)
    
    print(f"TRL implementation accuracy: {trl_accuracy:.4f}")
    print(f"PyTorch implementation accuracy: {pytorch_accuracy:.4f}")
    
    # Print some example responses
    print("\nExample responses:")
    for i in range(min(3, len(test_cases))):
        print(f"\nQuestion: {test_cases[i]['prompt'][1]['content']}")
        print(f"Gold answer: {test_cases[i]['answer']}")
        print(f"TRL response: {trl_results[i]['extracted_answer']}")
        print(f"PyTorch response: {pytorch_results[i]['extracted_answer']}")

if __name__ == "__main__":
    main() 