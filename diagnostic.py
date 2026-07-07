"""
Diagnostic script for SDPO training.
Checks:
1. Formatted prompts (p_text vs xo_text) — are they what we expect?
2. Tokenization alignment — do completion tokens line up correctly?
3. Logprob computation — are advantages sane?
4. Token-level signal — which tokens get reinforced/penalized?

Run on a single GPU:
  CUDA_VISIBLE_DEVICES=0 python diagnostic.py
"""

import json
import torch
import copy
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F


def load_data(jsonl_path: str, n: int = 3):
    ds = load_dataset("json", data_files=jsonl_path, split="train")
    return [ds[i] for i in range(n)]


def normalize_messages(messages):
    normalized = []
    for msg in messages:
        if "value" in msg and "content" not in msg:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            original_role = msg.get("from", "user")
            new_role = role_map.get(original_role, original_role)
            normalized.append({"role": new_role, "content": msg["value"]})
        else:
            normalized.append(msg)
    return normalized


def format_prompts(tokenizer, ex):
    """Format standard and hindsight prompts, return all 3 templates for comparison."""
    clean_prompt = normalize_messages(ex["prompt"])
    fb = ex["user_response"].get("value") or ex["user_response"].get("content")
    o = fb.strip()

    # Standard prompt (x)
    p_text = tokenizer.apply_chat_template(
        clean_prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    # Template A (v1): append block to last user message
    hist_a = copy.deepcopy(clean_prompt)
    hist_a[-1]["content"] += (
        "\n\n[HINDSIGHT CONTEXT]\n"
        "The following is a user response to your previous, insufficient attempt. "
        "Improve your response to the user prompt.\n"
        f"Future User Message: {o}"
    )
    xo_a = tokenizer.apply_chat_template(
        hist_a, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    # Template B (v2): add assistant message
    hist_b = clean_prompt[:]
    hist_b.append({
        "role": "assistant",
        "content": (
            "=== HINDSIGHT CONTEXT ===\n"
            "[The following is a future user message. Use this to guide your answer to the user prompt.]\n"
            f"{o}"
        )
    })
    xo_b = tokenizer.apply_chat_template(
        hist_b, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    # Template C (paper Table 1): append to last user message with paper wording
    hist_c = copy.deepcopy(clean_prompt)
    hist_c[-1]["content"] += (
        "\n\n=== HINDSIGHT CONTEXT ===\n"
        "The following is a future user message. "
        f"Use this to guide your answer to the user prompt: {o}"
    )
    xo_c = tokenizer.apply_chat_template(
        hist_c, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )

    completion_text = ex["completion"].get("value") or ex["completion"].get("content")
    completion_text = completion_text.rstrip() + tokenizer.eos_token

    return p_text, xo_a, xo_b, xo_c, completion_text, o


def compute_logprobs(model, tokenizer, context_text, completion_text, device):
    """Replicate _token_logps_of_given_y from the trainer."""
    old_pad_side = tokenizer.padding_side
    old_trunc_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    enc = tokenizer(
        [context_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        add_special_tokens=False,
    ).to(device)

    tokenizer.padding_side = old_pad_side
    tokenizer.truncation_side = old_trunc_side

    comp_enc = tokenizer(
        [completion_text],
        padding=True,
        truncation=True,
        max_length=2048,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    y_ids = comp_enc["input_ids"]       # (1, C)
    y_mask = comp_enc["attention_mask"]  # (1, C)

    input_ids = torch.cat([enc["input_ids"], y_ids], dim=1)
    attention_mask = torch.cat([enc["attention_mask"], y_mask], dim=1)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    logits = logits[:, :-1, :]
    labels = input_ids[:, 1:]

    seq_len_y = y_ids.size(1)
    logits_y = logits[:, -seq_len_y:, :]
    labels_y = labels[:, -seq_len_y:]

    pad_id = tokenizer.pad_token_id
    labels_y = labels_y.masked_fill(y_mask == 0, -100)

    B, C, V = logits_y.shape
    nll = F.cross_entropy(
        logits_y.reshape(B * C, V),
        labels_y.reshape(B * C),
        reduction="none",
        ignore_index=-100,
    ).reshape(B, C)

    logprobs = -nll  # (1, C)
    return logprobs[0], y_ids[0], y_mask[0]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--data", default="./data/wildfeedback/wildfeedback_interactions.jsonl")
    p.add_argument("--n", type=int, default=3, help="Number of examples to inspect")
    p.add_argument("--template", default="all", choices=["A", "B", "C", "all"])
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,         attn_implementation="sdpa"
    ).to(device).eval()

    print(f"pad_token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    print(f"eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")

    examples = load_data(args.data, args.n)

    for idx, ex in enumerate(examples):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {idx}")
        print(f"{'='*80}")

        p_text, xo_a, xo_b, xo_c, completion_text, user_followup = format_prompts(tokenizer, ex)

        # --- Show formatted prompts ---
        print(f"\n--- USER FOLLOW-UP (o) ---")
        print(user_followup[:300])

        print(f"\n--- STANDARD PROMPT (x) [last 500 chars] ---")
        print(p_text[-500:])

        print(f"\n--- COMPLETION (y) [first 200 chars] ---")
        print(completion_text[:200])

        templates = {"A": xo_a, "B": xo_b, "C": xo_c}
        if args.template != "all":
            templates = {args.template: templates[args.template]}

        for tname, xo_text in templates.items():
            print(f"\n--- TEMPLATE {tname} HINDSIGHT PROMPT [last 800 chars] ---")
            print(xo_text[-800:])

        # --- Tokenization check ---
        print(f"\n--- TOKENIZATION CHECK ---")
        p_tokens = tokenizer(p_text, add_special_tokens=False)["input_ids"]
        comp_tokens = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
        print(f"Standard prompt tokens: {len(p_tokens)}")
        print(f"Completion tokens: {len(comp_tokens)}")
        print(f"Last 5 prompt tokens: {p_tokens[-5:]} -> {[tokenizer.decode([t]) for t in p_tokens[-5:]]}")
        print(f"First 5 completion tokens: {comp_tokens[:5]} -> {[tokenizer.decode([t]) for t in comp_tokens[:5]]}")

        # Check boundary: does the last prompt token + first completion token make sense?
        boundary_text = tokenizer.decode(p_tokens[-3:] + comp_tokens[:3])
        print(f"Boundary (last 3 prompt + first 3 completion): {boundary_text!r}")

        # --- Logprob computation ---
        print(f"\n--- LOGPROB ANALYSIS ---")

        # Standard logprobs
        logps_x, y_ids, y_mask = compute_logprobs(model, tokenizer, p_text, completion_text, device)
        active = y_mask.bool()
        active_logps_x = logps_x[active]

        print(f"Standard logprobs: mean={active_logps_x.mean():.4f}, "
              f"min={active_logps_x.min():.4f}, max={active_logps_x.max():.4f}")

        for tname, xo_text in templates.items():
            logps_xo, _, _ = compute_logprobs(model, tokenizer, xo_text, completion_text, device)
            active_logps_xo = logps_xo[active]
            advantage = (logps_xo - logps_x)[active]

            print(f"\nTemplate {tname}:")
            print(f"  Hindsight logprobs: mean={active_logps_xo.mean():.4f}, "
                  f"min={active_logps_xo.min():.4f}, max={active_logps_xo.max():.4f}")
            print(f"  Advantage (A_i): mean={advantage.mean():.4f}, "
                  f"std={advantage.std():.4f}, "
                  f"min={advantage.min():.4f}, max={advantage.max():.4f}")
            print(f"  Positive advantage tokens: {(advantage > 0).sum()}/{active.sum()}")
            print(f"  Negative advantage tokens: {(advantage < 0).sum()}/{active.sum()}")

            # Show first 30 tokens with their advantages
            tok_ids = y_ids[active].tolist()
            lps_x = logps_x[active].tolist()
            lps_xo = logps_xo[active].tolist()
            advs = advantage.tolist()

            n_show = min(30, len(tok_ids))
            print(f"\n  First {n_show} tokens:")
            print(f"  {'idx':>4} | {'token':<20} | {'logp_x':>10} | {'logp_xo':>10} | {'advantage':>10}")
            print(f"  {'-'*70}")
            for i in range(n_show):
                tok_str = tokenizer.decode([tok_ids[i]]).replace("\n", "\\n")
                if len(tok_str) > 18:
                    tok_str = tok_str[:15] + "..."
                print(f"  {i:4d} | {tok_str:<20} | {lps_x[i]:10.4f} | {lps_xo[i]:10.4f} | {advs[i]:10.4f}")

            # Show last 10 tokens (including EOS)
            if len(tok_ids) > 30:
                print(f"\n  Last 10 tokens:")
                for i in range(max(0, len(tok_ids)-10), len(tok_ids)):
                    tok_str = tokenizer.decode([tok_ids[i]]).replace("\n", "\\n")
                    if len(tok_str) > 18:
                        tok_str = tok_str[:15] + "..."
                    print(f"  {i:4d} | {tok_str:<20} | {lps_x[i]:10.4f} | {lps_xo[i]:10.4f} | {advs[i]:10.4f}")

            # Check EOS token specifically
            eos_id = tokenizer.eos_token_id
            eos_positions = [i for i, t in enumerate(tok_ids) if t == eos_id]
            if eos_positions:
                print(f"\n  EOS token positions: {eos_positions}")
                for pos in eos_positions:
                    print(f"    pos {pos}: logp_x={lps_x[pos]:.4f}, logp_xo={lps_xo[pos]:.4f}, "
                          f"advantage={advs[pos]:.4f}")

    print(f"\n{'='*80}")
    print("DIAGNOSTIC COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
