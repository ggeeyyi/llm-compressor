import torch, os, random
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel
import argparse

def generate_dummy_lora(args):
    BASE = args.base_model
    DTYPE = torch.bfloat16
    device_map = "auto"

    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=DTYPE, device_map=device_map, trust_remote_code=True
    )

    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        bias=args.lora_bias, task_type=args.lora_task_type,
    )

    model = get_peft_model(model, lora_cfg)

    # Create local directory with rank in name
    INIT_DIR = f"{args.lora_path}_r{args.lora_rank}"
    os.makedirs(INIT_DIR, exist_ok=True)
    model.save_pretrained(INIT_DIR)
    tok.save_pretrained(INIT_DIR)
    
    print(f"Model saved locally to: {INIT_DIR}")
    
    # Push to HuggingFace with rank in repository name
    if args.push_to_hub:
        huggingface_path = f'GY2233/{INIT_DIR}'
        print(f"Pushing model to HuggingFace: {huggingface_path}")
        
        try:
            # Push model to HuggingFace
            model.push_to_hub(huggingface_path)
            # Push tokenizer to the same repository
            tok.push_to_hub(huggingface_path)
            print(f"Successfully pushed model to: https://huggingface.co/{huggingface_path}")
        except Exception as e:
            print(f"Error pushing to HuggingFace: {e}")
            print("Make sure you have the correct HuggingFace token and permissions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")
    parser.add_argument("--lora_bias", type=str, default="none")
    parser.add_argument("--lora_task_type", type=str, default="CAUSAL_LM")
    parser.add_argument("--lora_path", type=str, default="qwen25_3b_lora_init_default")
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()
    generate_dummy_lora(args)