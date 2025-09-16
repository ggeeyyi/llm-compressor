from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import argparse
import os
import json
import pandas as pd
import torch
from llmcompressor.utils import dispatch_for_generation

def save_results(args, results):
    print(f"Debug: results type: {type(results)}")
    print(f"Debug: results content: {results}")
    if isinstance(results, dict):
        print(f"Debug: results keys: {list(results.keys())}")
        print(f"Debug: results values: {list(results.values())}")
    df = pd.DataFrame([results])  # Wrap the dictionary in a list
    safe_model_name = args.model.replace('/', '_').replace('\\', '_')
    df.to_csv(f"huggingface_results_{safe_model_name}_add_lora_{args.add_lora}.csv", index=False)

def padding_tokens(text: str, tok, max_tokens: int) -> str:
    ids = tok.encode(text)
    if len(ids) >= max_tokens:
        ids = ids[:max_tokens]
    else:
        ids = ids + [tok.eos_token_id] * (max_tokens - len(ids))
    return tok.decode(ids, skip_special_tokens=True)

def prepare_prompts(args, tok, prefill_length: int):
    # prompts = ["Let $ABCD$ be a tetrahedron such that $AB=CD= \sqrt{41}$, $AC=BD= \sqrt{80}$, and $BC=AD= \sqrt{89}$. There exists a point $I$ inside the tetrahedron such that the distances from $I$ to each of the faces of the tetrahedron are all equal. This distance can be written in the form $\frac{m \sqrt n}{p}$, where $m$, $n$, and $p$ are positive integers, $m$ and $p$ are relatively prime, and $n$ is not divisible by the square of any prime. Find $m+n+p$."]
    prompts = ["Write a poem about a cat." * 1000]
    if args.batch_size != 1:
        prompts = prompts * args.batch_size
    prompts = [padding_tokens(prompt, tok, prefill_length) for prompt in prompts]
    return prompts

def test_efficiency(args):
    # Load model and tokenizer following the pattern from quantize_nvfp4.py
    MODEL_ID = args.model
    if 'qwen' in MODEL_ID.lower():
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    elif 'llama' in MODEL_ID.lower():
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype="auto", trust_remote_code=True
        )
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    dispatch_for_generation(model)
    # Set pad token if not present
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = prepare_prompts(args, tok, prefill_length=args.prefill_length)
    
    # test prefill time
    start_prefill_time = time.time()
    # Tokenize all prompts for prefill test
    input_ids_list = []
    for prompt in prompts:
        input_ids = tok(prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to(model.device)
        input_ids_list.append(input_ids)
    
    # Generate 1 token for prefill test
    prefill_outputs = []
    for input_ids in input_ids_list:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tok.eos_token_id
            )
        prefill_outputs.append(output)
    
    end_prefill_time = time.time()
    elapsed_prefill_time = end_prefill_time - start_prefill_time
    prefill_tokens_generated = sum(input_ids.shape[1] for input_ids in input_ids_list)
    tokens_per_second = prefill_tokens_generated / elapsed_prefill_time

    print(f"Batch size: {len(prefill_outputs)}")
    print(f"Total prefill tokens: {prefill_tokens_generated}")
    print(f"Prefill time taken: {elapsed_prefill_time:.4f} seconds")
    print(f"Prefill tokens per second: {tokens_per_second:.2f}")
    # test total time
    start_total_time = time.time()
    # Generate full sequences
    total_outputs = []
    for input_ids in input_ids_list:
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=args.decode_length,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tok.eos_token_id
            )
        total_outputs.append(output)
    
    end_total_time = time.time()
    elapsed_total_time = end_total_time - start_total_time
    # Calculate total generated tokens across all requests in the batch
    total_tokens_generated = sum(output.shape[1] - input_ids.shape[1] for output, input_ids in zip(total_outputs, input_ids_list))
    tokens_per_second_total = total_tokens_generated / elapsed_total_time
    
    print(f"\nTotal generation results:")
    print(f"Total tokens generated: {total_tokens_generated}")
    print(f"Total time taken: {elapsed_total_time:.4f} seconds")
    print(f"Total tokens per second: {tokens_per_second_total:.2f}")
    
    # Calculate decode-only metrics
    elapsed_decode_time = elapsed_total_time - elapsed_prefill_time
    decode_tokens_generated = total_tokens_generated
    tokens_per_second_decode = decode_tokens_generated / elapsed_decode_time if elapsed_decode_time > 0 else 0
    
    print(f"\nDecode-only results:")
    print(f"Decode tokens generated: {decode_tokens_generated}")
    print(f"Decode time taken: {elapsed_decode_time:.4f} seconds")
    print(f"Decode tokens per second: {tokens_per_second_decode:.2f}")
    return {
        "batch_size": args.batch_size,
        "prefill_length": args.prefill_length,
        "decode_length": args.decode_length,
        "add_lora": args.add_lora,
        "lora_path": args.lora_path,
        "prefill_time": elapsed_prefill_time,
        "prefill_tokens_per_second": tokens_per_second,
        "total_time": elapsed_total_time,
        "total_tokens_per_second": tokens_per_second_total,
        "decode_time": elapsed_decode_time,
        "decode_tokens_per_second": tokens_per_second_decode,
        "total_tokens_generated": total_tokens_generated,
        "prefill_tokens_generated": prefill_tokens_generated,
        "decode_tokens_generated": decode_tokens_generated,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="quantized_model/Qwen2.5-3B-NVFP4A16")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prefill_length", type=int, default=1024)
    parser.add_argument("--decode_length", type=int, default=128)
    parser.add_argument("--add_lora", action="store_true")
    parser.add_argument("--lora_path", type=str, default="bunnycore/qwen-2.5-3b-lora_model")
    parser.add_argument("--test_mode", type=str, default="mixed", choices=["prefill", "decode", "mixed"])
    args = parser.parse_args()
    results = test_efficiency(args)
    save_results(args, results)