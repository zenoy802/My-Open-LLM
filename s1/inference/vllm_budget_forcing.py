from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import os
os.environ['HF_HOME'] = '/root/autodl-tmp/.cache/'

# Decide on a token limit for thinking; As the model's max tokens is 32768, 32000 usually ensures there is enough space for the model to still answer
MAX_TOKENS_THINKING = 10000
# Decide how often to ignore end-of-thinking token
NUM_IGNORE = 1

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
    skip_special_tokens=False,
    temperature=0.0,
)

# For the exact raspberry sample in the paper, change
# model to `qfq/1k_qr_bt_dm_po_steps` (an earlier version of s1)
# & prompt to `How many r in raspberry?`
prompts = [
    "How many r in raspberry",
]

for i, p in enumerate(prompts):
    prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + p + "<|im_end|>\n<|im_start|>assistant\n"
    stop_token_ids = tok("<|im_start|><|im_end|>")["input_ids"]
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS_THINKING,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=0.0,
    )
    prompt += "<|im_start|>think"
    o = model.generate(
        prompt,
        sampling_params=sampling_params
    )
    ignore_str = "Wait"
    max_tokens_thinking_tmp = MAX_TOKENS_THINKING
    # Num of times to skip stop token
    for i in range(NUM_IGNORE):
        max_tokens_thinking_tmp -= len(o[0].outputs[0].token_ids)
        prompt += o[0].outputs[0].text + ignore_str
        sampling_params = SamplingParams(
            max_tokens=max_tokens_thinking_tmp,
            min_tokens=1,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=False,
            temperature=0.0,
        )
        o = model.generate(
            prompt,
            sampling_params=sampling_params
        )
    ### Final answer ###
    prompt += o[0].outputs[0].text # You can also append "Final Answer:" here like we do for some evaluations to prevent the model from just continuing to reason in its answer when early exiting
    stop_token_ids = tok("<|im_end|>")["input_ids"]
    sampling_params = SamplingParams(
        max_tokens=12000,
        min_tokens=0,
        stop_token_ids=stop_token_ids,
        skip_special_tokens=False,
        temperature=0.0,
    )
    o = model.generate(
        prompt,
        sampling_params=sampling_params,
    )
    print("With budget forcing:")
    print(prompt + o[0].outputs[0].text)