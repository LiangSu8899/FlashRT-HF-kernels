# Source Sync

Synced from `official/FlashRT`:

- `csrc/quantize/qkv_split_norm_rope_bf16.cu`: packed QKV split, Q/K RMSNorm,
  and RoPE math.
- `csrc/quantize/qkv_split_norm_rope_bf16.cuh`: public pointer-level contract.
- `csrc/kernels/qwen3_qkv_post_proc.cu`: batched per-head Q/K RMSNorm and
  rotate-half RoPE math, plus decode Q staging and KV cache-write math; public
  entry points were renamed to generic APIs for this package.
- `csrc/kernels/qwen3_qkv_post_proc.cuh`: decode pointer-level contract.
- `csrc/kernels/rope.cu`: GQA sequence `qkv_split_rope` /
  `qkv_split_rope_kvcache` adjacent-pair RoPE and KV-cache write math.
- `csrc/kernels/rope.cuh`: source declaration for the GQA sequence cache
  contract.
- `csrc/kernels/lingbot_rope_qkv.cu`: packed GQA/equal-head QKV bias,
  split-half RoPE, and direct Q/K/V workspace writes. The public package uses
  a model-agnostic name and accepts both half-width and full-width RoPE tables.

Local adaptation:

- Raw pointer APIs were converted to Tensor-based `torch.ops` bindings.
- The current package exposes no-bias Q/K, bias+Q/K/V, bias+cat-workspace,
  three-segment VLA joint workspace APIs, per-head GQA norm/RoPE/staging,
  decode Q staging, direct KV-write, device-position KV-write, and GQA
  sequence split/RoPE/KV-cache write.
- The fused bias+RoPE path preserves the original native arithmetic and
  packed-QKV layout while generalizing the binding across Q/KV head counts and
  head dimensions 64, 72, 80, and 128 used by current model pipelines.
- The FP16-output form is synchronized from the same native LingBot QKV
  template. It avoids a standalone BF16-to-FP16 conversion before an FP16
  attention core while retaining BF16 projected inputs and biases.
- The per-head GQA sequence API extends the upstream batched Q/K math to a
  single Tensor contract that consumes packed QKV and writes Q/K/V workspaces.
  It keeps the honest upstream `head_dim == 128` constraint while allowing
  unequal Q and KV head counts.
- The decode functions use generic public names and explicit shape validation.
  They keep the upstream `head_dim == 128` contract instead of pretending to be
  arbitrary-head-dimension kernels.
- The GQA sequence cache API keeps the upstream interleaved
  `[cos0, sin0, ...]` RoPE table contract, but exposes a Tensor layout:
  `packed_qkv=(B,S,(QH+2*KVH)*HD)`, `q_out=(B,S,QH,HD)`, and
  `k_cache/v_cache=(B,max_S,KVH,HD)`.
- The ROCm implementation is an architecture port of that public GQA cache
  contract, not a second API. It uses HIP BF16 scalar and aligned 16-byte
  vector paths, preserves the adjacent-pair RoPE arithmetic and dense cache
  layout, and is built only for `gfx950`. Other package symbols remain
  CUDA-only until each receives its own correctness and performance evidence.
