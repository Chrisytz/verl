# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import warnings
import re
import argparse
from typing import Dict, Optional

from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Environment variable setup for distributed VLLM (if applicable)
os.environ.setdefault("RANK", os.getenv("RANK", "0"))
os.environ.setdefault("LOCAL_RANK", os.getenv("LOCAL_RANK", "0"))
os.environ.setdefault("WORLD_SIZE", os.getenv("WORLD_SIZE", "1"))
os.environ.setdefault("MASTER_ADDR", os.getenv("MASTER_ADDR", "localhost"))
os.environ.setdefault("MASTER_PORT", os.getenv("MASTER_PORT", "8269"))

# Constants
QWEN_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
GSM8K_DATASET = "openai/gsm8k"
PROMPT_VERL = "verl"
PROMPT_QWEN = "qwen"
METHOD_STRICT = "strict"
METHOD_FLEXIBLE = "flexible"

def set_tokenizer_padding(tokenizer: AutoTokenizer) -> None:
    """Sets pad_token_id and pad_token for the tokenizer if they are None."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        warnings.warn(f"tokenizer.pad_token_id is None. Set to {tokenizer.eos_token_id}", stacklevel=1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn(f"tokenizer.pad_token is None. Set to {tokenizer.eos_token}", stacklevel=1)

def load_model_tokenizer() -> AutoTokenizer:
    """Loads and configures the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=False)
    set_tokenizer_padding(tokenizer)
    return tokenizer

def format_message_to_prompt(prompt_type: str, message: Dict[str, str], few_shot_prompt="") -> str:
    """Formats a message dictionary into a raw prompt string based on the prompt type."""
    if prompt_type == PROMPT_VERL:
        return (
            'You are Qwen, created by Alibaba Cloud. You are a helpful assistant\n'
            + message["question"]
            + '\nLet\'s think step by step and output the final answer after "####".\n'
        )
    elif prompt_type == PROMPT_QWEN:
        return (
            few_shot_prompt
            + "\nQuestion: "
            + message["question"]
            + "\nLet's think step by step\n"
        )
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

def extract_solution(solution_str, matching_token="####", method=METHOD_STRICT) -> Optional[str]:
    """Extracts numerical solution from model response"""
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        solution = re.search(f"{matching_token} (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split(f"{matching_token} ")[1].replace(",", "").replace("$", "")
            if final_answer[-1] == ".":
                final_answer = final_answer[:-1]
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    if final_answer is None:
        return None

    try:
        if '.' in final_answer:
            return float(final_answer)
        else:
            return int(final_answer)
    except ValueError:
        print(f"Could not convert extracted string '{final_answer}' to a number.")
        return None

def compute_score(solution_str, ground_truth, matching_token="####", method=METHOD_STRICT, format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str, matching_token=matching_token, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate language model on GSM8k dataset.")
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        choices=[PROMPT_VERL, PROMPT_QWEN],
        help=f"Prompt format to use. Either '{PROMPT_VERL}' or '{PROMPT_QWEN}'.",
        default=PROMPT_VERL,
    )
    parser.add_argument(
        "-m",
        "--extract-solution-method",
        type=str,
        choices=[METHOD_STRICT, METHOD_FLEXIBLE],
        help=f"Method to extract numerical answer from response. Either '{METHOD_STRICT}' or '{METHOD_FLEXIBLE}'.",
        default=METHOD_STRICT,
    )
    parser.add_argument(
        "-f",
        "--file-path",
        type=str,
        help="Optional path to few shot prompt."
    )
    parser.add_argument
    
    args = parser.parse_args()
    
    few_shot_prompt = ""
    if args.file_path:
        few_shot_prompt = open(args.file_path).read()
    
    tokenizer = load_model_tokenizer()
    ds = load_dataset(GSM8K_DATASET, "main", split="test")

    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        enable_sleep_mode=False,
        tensor_parallel_size=1,
        distributed_executor_backend="external_launcher",
        dtype='bfloat16',
        enforce_eager=False,
        gpu_memory_utilization=0.4,
        disable_custom_all_reduce=True,
        disable_mm_preprocessor_cache=False,
        skip_tokenizer_init=False,
        max_model_len=1536,
        disable_log_stats=True,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        trust_remote_code=False,
        seed=0,
    )

    vllm_inputs = []
    ground_truths = []
    for message in ds:
        raw_prompt = format_message_to_prompt(args.prompt, message, few_shot_prompt)
        model_inputs = tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        vllm_inputs.append(model_inputs["input_ids"][0].tolist())
        ground_truths.append(extract_solution(message["answer"], method=METHOD_STRICT)) # Ground truth extraction is always strict

    sampling_params_json = {
        "n": 1,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
        "min_p": 0.0,
        "seed": None,
        "stop": [],
        "stop_token_ids": [],
        "bad_words": [],
        "include_stop_str_in_output": False,
        "ignore_eos": False,
        "max_tokens": 512,
        "min_tokens": 0,
        "logprobs": 0,
        "prompt_logprobs": None,
        "skip_special_tokens": True,
        "spaces_between_special_tokens": True,
        "truncate_prompt_tokens": None,
        "guided_decoding": None,
        "extra_args": None
        }
    
    sampling_params = SamplingParams(**sampling_params_json)

    outputs = llm.generate(
        prompt_token_ids=vllm_inputs,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    matching_token = "####" if args.prompt == PROMPT_VERL else "The answer is"
    total_possible_score = 0
    total_score = 0

    for i, (output, ground_truth) in enumerate(zip(outputs, ground_truths)):
        generated_text = output.outputs[0].text
        score = compute_score(generated_text, ground_truth,
                                     matching_token=matching_token,
                                     method=args.extract_solution_method)

        print(f"Sample {i+1}:")
        print(f"Generated text: {generated_text!r}")
        print(f"Ground truth: {ground_truth!r}")
        print(f"Score: {score}")

        total_possible_score += 1
        total_score += score

    accuracy = total_score/ total_possible_score
    print(f"Final accuracy: {accuracy:.4f} ({total_score}/{total_possible_score})")

if __name__ == "__main__":
    main()