import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import os
import sys
from verl.utils.tokenizer import hf_tokenizer
from verl.utils.model import update_model_config

def convert_checkpoint(base_model: str, checkpoint_path: str, output_dir: str):
    """
    Loads a base model, applies a fine-tuned state_dict, and saves the result
    in a vLLM-compatible Hugging Face format.
    """

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)

    tokenizer = hf_tokenizer(base_model, trust_remote_code=True)

    override_config_kwargs = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    update_model_config(config, override_config_kwargs=override_config_kwargs)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
        config=config,
        trust_remote_code=True,
    )

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at '{checkpoint_path}'", file=sys.stderr)
        sys.exit(1)

    state_dict = torch.load(checkpoint_path)

    model.load_state_dict(state_dict)

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Conversion complete! Model saved to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a fine-tuned PyTorch state_dict to a full Hugging Face model directory."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="The Hugging Face model ID of the base model (e.g., 'Qwen/Qwen2.5-0.5B-Instruct')."
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the saved model."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the final, vLLM-compatible model."
    )

    args = parser.parse_args()
    convert_checkpoint(args.base_model, args.weights, args.output_dir)