# Chunked KL Divergence for Large-Vocabulary Distillation

## Problem

Computing `KL(teacher || student)` over a 100K+ token vocabulary materializes a `[B, seq_len, vocab_size]` tensor. For Qwen3-8B (vocab=151K) with 2048 completion tokens, this is ~1.16 GB per sample -- causes OOM even on H100 80GB.

## Solution

Chunk the KL computation over the sequence dimension. Peak memory drops from `O(B * C * V)` to `O(B * chunk_size * V)`.

```python
KL_CHUNK = 128
per_token_kl = torch.zeros(B, C, device=device)
for i in range(0, C, KL_CHUNK):
    j = min(i + KL_CHUNK, C)
    p_teacher = F.softmax(logits_teacher[:, i:j, :], dim=-1).detach()
    log_p_student = F.log_softmax(logits_student[:, i:j, :], dim=-1)
    per_token_kl[:, i:j] = F.kl_div(
        log_p_student, p_teacher, reduction='none'
    ).sum(dim=-1)
```

With chunk_size=128: peak is `1 * 128 * 151K * 4 bytes = 73 MB` instead of 1.16 GB.

## References

### Implementations

| Project | Approach | Link |
|---|---|---|
| Liger Kernel (LinkedIn) | Fused linear + chunked loss, operates on hidden states not logits. ~80% memory reduction. | [github.com/linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel) - `fused_linear_distillation.py` |
| Flash Linear Attention | Triton kernel with online softmax, doubly chunked (sequence in Python, vocab in Triton). | [github.com/fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention) - `fla/modules/fused_kl_div.py` |
| HuggingFace TRL | `DistillationTrainer` routes through Liger kernel (`use_liger_kernel=True`); also supports top-K sparse approximation. | [github.com/huggingface/trl](https://github.com/huggingface/trl) - `trl/experimental/distillation/distillation_trainer.py` |
| Arcee DistillKit | Automatic chunking for logit-based distillation. | [github.com/arcee-ai/DistillKit](https://github.com/arcee-ai/DistillKit) |

### Papers

- "Liger Kernel: Efficient Triton Kernels for LLM Training" - [arXiv:2410.10989](https://arxiv.org/abs/2410.10989) (2024)
- "Cut Your Losses in Large-Vocabulary Language Models" - Apple, ICLR 2025. Foundation work on chunked cross-entropy.
- "DistiLLM: Towards Streamlined Distillation for Large Language Models" - ICML 2024. Skew-KL for LLM distillation. [arXiv:2402.03898](https://arxiv.org/abs/2402.03898)
- "On-Policy Distillation of Language Models" (GKD) - ICLR 2024. Foundation for TRL's distillation. [arXiv:2306.13649](https://arxiv.org/abs/2306.13649)

### Discussion

- [PyTorch forum: Using Tensor subclasses for chunked loss and backprop](https://discuss.pytorch.org/t/using-tensor-subclasses-for-chunked-loss-and-backprop/212525) - canonical pattern description

## Future

More aggressive optimizations if needed:
- Fuse lm_head projection into the chunk loop (Liger/FLA approach) -- operate on hidden states, never materialize full logits
- Triton kernel with online softmax (FLA approach) -- O(1) extra memory
- Top-K sparse KL (TRL approach) -- approximate but avoids full vocab entirely
