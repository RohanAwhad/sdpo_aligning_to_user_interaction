"""
On-policy SDFT with vLLM rollouts.

Outer loop:
  1. Generate completions via vLLM batch inference (first half of GPUs)
  2. Train 1 epoch on those completions (second half of GPUs, accelerate launch)
  3. Save checkpoint, repeat

Usage:
  export WANDB_MODE=online
  export WANDB_ENTITY=ronny21
  export WANDB_PROJECT=sdpo-amortize
  export WANDB_NAME=on-policy-vllm-run-1
  export TOKENIZERS_PARALLELISM=false
  export OUTPUT_DIR=./output/on-policy-vllm-run-1

  python offline_sdpo/run_on_policy_vllm.py \
    --base_model Qwen/Qwen3-8B \
    --train_jsonl /path/to/train.jsonl \
    --num_epochs 10 \
    --num_gpus 8 \
    --learning_rate 2e-6 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr_scheduler_type constant \
    --logit_loss \
    --logit_loss_topk 16 \
    --gen_max_new_tokens 2048 \
    --gen_temperature 0.7
"""

import argparse
import json
import os
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser()
    # Model
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--train_jsonl", type=str, required=True)

    # GPU split
    p.add_argument("--num_gpus", type=int, required=True,
                   help="Total GPUs (>= 2, divisible by 2). First half for vLLM, second half for training.")

    # Outer loop
    p.add_argument("--num_epochs", type=int, default=10,
                   help="Number of outer epochs (each = 1 vLLM rollout + 1 training epoch)")

    # Training args (passed through to accelerate launch main_offline_sdpo.py)
    p.add_argument("--learning_rate", type=float, default=2e-6)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr_scheduler_type", type=str, default="constant",
                   choices=["cosine", "constant", "linear"])
    p.add_argument("--save_steps", type=int, default=9999,
                   help="Save steps within each epoch (default: effectively off)")
    p.add_argument("--logit_loss", action="store_true")
    p.add_argument("--logit_loss_topk", type=int, default=16)

    # vLLM rollout args
    p.add_argument("--gen_max_new_tokens", type=int, default=2048)
    p.add_argument("--gen_temperature", type=float, default=0.7)
    p.add_argument("--vllm_python", type=str, default=".venv_vllm/bin/python",
                   help="Path to python in vLLM venv (default: .venv_vllm/bin/python)")

    # Internal: rollout subprocess mode
    p.add_argument("--_rollout", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_rollout_model", type=str, help=argparse.SUPPRESS)
    p.add_argument("--_rollout_input", type=str, help=argparse.SUPPRESS)
    p.add_argument("--_rollout_output", type=str, help=argparse.SUPPRESS)

    args = p.parse_args()

    if not args._rollout:
        if args.num_gpus < 2 or args.num_gpus % 2 != 0:
            p.error("--num_gpus must be >= 2 and divisible by 2")

    return args


# ---------------------------------------------------------------------------
# Rollout subprocess entry point
# ---------------------------------------------------------------------------

def _run_rollout(args):
    """Called in subprocess with CUDA_VISIBLE_DEVICES set to vLLM GPUs."""
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    from vllm import LLM, SamplingParams

    model_path = args._rollout_model
    input_path = args._rollout_input
    output_path = args._rollout_output

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tp_size = len(visible.split(",")) if visible else 1

    with open(input_path) as f:
        records = [json.loads(line) for line in f]

    print(f"[vLLM] Loading {model_path} (tp={tp_size})...")
    llm = LLM(model=model_path, tensor_parallel_size=tp_size, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=args.gen_temperature,
        max_tokens=args.gen_max_new_tokens,
    )

    # Build prompts using same chat template as training
    prompts = []
    for rec in records:
        messages = _normalize_messages(rec["prompt"])
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompts.append(prompt)

    print(f"[vLLM] Generating {len(prompts)} completions...")
    outputs = llm.generate(prompts, sampling_params)

    for rec, output in zip(records, outputs):
        rec["completion"] = {"from": "gpt", "value": output.outputs[0].text}

    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[vLLM] Wrote {len(records)} completions to {output_path}")


def _normalize_messages(messages):
    """from/value -> role/content"""
    normalized = []
    for msg in messages:
        if "value" in msg and "content" not in msg:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            role = role_map.get(msg.get("from", "user"), msg.get("from", "user"))
            normalized.append({"role": role, "content": msg["value"]})
        else:
            normalized.append(msg)
    return normalized


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _gpu_split(num_gpus):
    """Returns (vllm_gpu_ids, train_gpu_ids) as comma-separated strings."""
    half = num_gpus // 2
    vllm_ids = ",".join(str(i) for i in range(half))
    train_ids = ",".join(str(i) for i in range(half, num_gpus))
    return vllm_ids, train_ids


def run_rollout_subprocess(model_path, input_jsonl, output_jsonl, vllm_gpus, args):
    """Spawn a subprocess to generate rollouts with vLLM on the first half of GPUs."""
    script_path = os.path.abspath(__file__)
    vllm_python = os.path.abspath(args.vllm_python)
    cmd = [
        vllm_python, script_path,
        "--_rollout",
        "--_rollout_model", model_path,
        "--_rollout_input", input_jsonl,
        "--_rollout_output", output_jsonl,
        "--gen_max_new_tokens", str(args.gen_max_new_tokens),
        "--gen_temperature", str(args.gen_temperature),
        # These are required by parse_args but unused in rollout mode
        "--base_model", args.base_model,
        "--train_jsonl", args.train_jsonl,
        "--num_gpus", str(args.num_gpus),
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = vllm_gpus

    print(f"[Orchestrator] Rollout subprocess: CUDA_VISIBLE_DEVICES={vllm_gpus}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Rollout subprocess failed with exit code {result.returncode}")


def run_training_subprocess(model_path, train_jsonl, output_dir, train_gpus, args, epoch_idx):
    """Spawn accelerate launch to train 1 epoch on the second half of GPUs."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "main_offline_sdpo.py")
    num_train_gpus = len(train_gpus.split(","))

    # Use training venv python (sibling of this script's venv)
    repo_root = os.path.dirname(script_dir)
    train_python = os.path.join(repo_root, ".venv", "bin", "python")

    cmd = [
        train_python, "-m", "accelerate.commands.launch",
        "--num_processes", str(num_train_gpus),
        "--mixed_precision", "bf16",
        train_script,
        "--base_model", model_path,
        "--train_jsonl", train_jsonl,
        "--learning_rate", str(args.learning_rate),
        "--batch_size", str(args.batch_size),
        "--grad_accum", str(args.grad_accum),
        "--num_epochs", "1",
        "--save_steps", str(args.save_steps),
        "--lr_scheduler_type", args.lr_scheduler_type,
    ]
    if args.logit_loss:
        cmd += ["--logit_loss", "--logit_loss_topk", str(args.logit_loss_topk)]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = train_gpus

    epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch_idx}")
    env["OUTPUT_DIR"] = epoch_output_dir

    run_name = os.environ.get("WANDB_NAME", "on-policy-vllm")
    env["WANDB_NAME"] = f"{run_name}_epoch{epoch_idx}"

    print(f"[Orchestrator] Training subprocess: CUDA_VISIBLE_DEVICES={train_gpus}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Training subprocess failed with exit code {result.returncode}")

    return os.path.join(epoch_output_dir, "final_model")


def main():
    args = parse_args()

    # Internal rollout mode — called by orchestrator as subprocess
    if args._rollout:
        _run_rollout(args)
        return

    output_dir = os.environ.get("OUTPUT_DIR", "./output/on_policy_vllm")
    os.makedirs(output_dir, exist_ok=True)

    vllm_gpus, train_gpus = _gpu_split(args.num_gpus)

    print(f"GPU split: vLLM=[{vllm_gpus}]  Training=[{train_gpus}]")
    print(f"Base model: {args.base_model}")
    print(f"Train JSONL: {args.train_jsonl}")
    print(f"Outer epochs: {args.num_epochs}")
    print(f"Logit loss: {args.logit_loss}" + (f" (top-k={args.logit_loss_topk})" if args.logit_loss else ""))

    current_model = args.base_model

    for epoch in range(args.num_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{args.num_epochs}")
        print(f"Model: {current_model}")
        print(f"{'='*60}\n")

        # Step 1: Generate rollouts (subprocess, first half GPUs)
        epoch_jsonl = os.path.join(output_dir, f"rollouts_epoch_{epoch}.jsonl")
        run_rollout_subprocess(current_model, args.train_jsonl, epoch_jsonl, vllm_gpus, args)
        print(f"[Data] Rollouts written to {epoch_jsonl}")

        # Step 2: Train 1 epoch (subprocess, second half GPUs)
        current_model = run_training_subprocess(
            model_path=current_model,
            train_jsonl=epoch_jsonl,
            output_dir=output_dir,
            train_gpus=train_gpus,
            args=args,
            epoch_idx=epoch,
        )
        print(f"[Train] Checkpoint saved to {current_model}")

    print(f"\nDone. Final model: {current_model}")


if __name__ == "__main__":
    main()
