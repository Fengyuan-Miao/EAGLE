import argparse
import json
import os
import sys
import time
from typing import Dict, List

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def parse_dtype(name: str):
    import torch

    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_prompts(path: str) -> List[Dict]:
    with open(path) as fin:
        return json.load(fin)


def get_input_device(model):
    if hasattr(model, "base_model"):
        model = model.base_model
    return next(model.parameters()).device


def sync_cuda():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from eagle.model.ea_model import EaModel

    dtype = parse_dtype(args.dtype)
    if args.decode_mode == "vanilla_ar":
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_path,
            use_fast=False,
            trust_remote_code=args.trust_remote_code,
        )
        model.eval()
        return model, tokenizer

    model = EaModel.from_pretrained(
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
        use_eagle3=True,
        spec_mode=args.spec_mode,
        enable_profile=args.enable_profile,
        profile_output=args.profile_output,
        opt_tree_budget=args.opt_tree_budget,
        opt_tree_overexpand_factor=args.opt_tree_overexpand_factor,
        opt_tree_score=args.opt_tree_score,
        ddd_max_depth=args.ddd_max_depth,
        ddd_checkpoints=args.ddd_checkpoints,
        ddd_thresholds=args.ddd_thresholds,
    )
    model.eval()
    return model, model.get_tokenizer()


def generate_one(model, tokenizer, prompt: str, args):
    import torch

    with torch.inference_mode():
        encoded = tokenizer([prompt], return_tensors="pt")
        input_ids = encoded.input_ids.to(get_input_device(model))
        prompt_len = input_ids.shape[-1]

        sync_cuda()
        start = time.perf_counter()
        if args.decode_mode == "vanilla_ar":
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 1e-5,
            }
            if args.temperature > 1e-5:
                generation_kwargs["temperature"] = args.temperature
                generation_kwargs["top_p"] = args.top_p
            output_ids = model.generate(input_ids, **generation_kwargs)
            iterations = None
        else:
            output_ids, generated_count, iterations = model.eagenerate(
                input_ids,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.sampling_top_k,
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
                log=True,
            )
        sync_cuda()
        wall_time = time.perf_counter() - start
        generated_tokens = int(output_ids.shape[-1] - prompt_len)
        text = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    return {
        "prompt_len": int(prompt_len),
        "generated_tokens": generated_tokens,
        "wall_time_s": wall_time,
        "tokens_per_s": generated_tokens / wall_time if wall_time > 0 else 0.0,
        "iterations": iterations,
        "text": text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-mode", choices=["vanilla_ar", "eagle"], default="eagle")
    parser.add_argument("--spec-mode", choices=["baseline", "opt_tree", "ddd", "opt_tree_ddd"], default="baseline")
    parser.add_argument("--base-model-path", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--ea-model-path", default="/home/miaofy/models/Qwen3-1.7B_eagle3")
    parser.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "prompts.json"))
    parser.add_argument("--output", default="results/benchmark.jsonl")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--sampling-top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--total-token", type=int, default=32)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--enable-profile", action="store_true")
    parser.add_argument("--profile-output")
    parser.add_argument("--opt-tree-budget", type=int)
    parser.add_argument("--opt-tree-overexpand-factor", type=float, default=2.0)
    parser.add_argument("--opt-tree-score", default="path_prob")
    parser.add_argument("--ddd-max-depth", type=int)
    parser.add_argument("--ddd-checkpoints", default="5,7,9")
    parser.add_argument("--ddd-thresholds", default="-6,-8,-10")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    model, tokenizer = load_model(args)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.decode_mode == "vanilla_ar":
        mode = "vanilla_ar"
    elif args.spec_mode == "baseline":
        mode = "eagle_baseline"
    else:
        mode = args.spec_mode
    run_id = args.run_id or f"{mode}_{int(time.time())}"

    with open(args.output, "w") as fout:
        for prompt in prompts:
            import torch

            torch.manual_seed(args.seed)
            result = generate_one(model, tokenizer, prompt["prompt"], args)
            profile_summary = getattr(model, "last_profile_summary", {}) if args.decode_mode == "eagle" else {}
            record = {
                "run_id": run_id,
                "mode": mode,
                "model": args.base_model_path,
                "draft_model": args.ea_model_path if args.decode_mode == "eagle" else "",
                "prompt_id": prompt["id"],
                "category": prompt.get("category", ""),
                "prompt_len": result["prompt_len"],
                "max_new_tokens": args.max_new_tokens,
                "generated_tokens": result["generated_tokens"],
                "wall_time_s": result["wall_time_s"],
                "tokens_per_s": result["tokens_per_s"],
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
                "iterations": result["iterations"],
                **profile_summary,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()


if __name__ == "__main__":
    main()
