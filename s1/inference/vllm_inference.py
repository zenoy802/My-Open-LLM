from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import os
os.environ['HF_HOME'] = '/root/autodl-tmp/.cache/'

model = LLM(
    # "simplescaling/s1-32B",
    "Qwen/Qwen2.5-32B-Instruct",
    tensor_parallel_size=1,
    max_model_len=12000,
)
# tok = AutoTokenizer.from_pretrained("simplescaling/s1-32B")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", timeout=30)

stop_token_ids = tok("<|im_end|>")["input_ids"]

sampling_params = SamplingParams(
    max_tokens=12000,
    min_tokens=0,
    stop_token_ids=stop_token_ids,
)

prompt = "How many r in raspberry"
prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"

o = model.generate(prompt, sampling_params=sampling_params)
print(o[0].outputs[0].text)