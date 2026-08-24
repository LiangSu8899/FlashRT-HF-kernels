# flashrt-qkv-cache-rope

Tensor-facing FlashRT QKV split, Q/K RMSNorm, and RoPE kernels for Hugging Face
`kernels`.

This package targets VLA, VLM, and video-model attention staging where the
attention core itself can be provided by FA2/FA3/SDPA, but the QKV postprocess
chain is still expensive in PyTorch:

```text
packed QKV -> split Q/K -> RMSNorm Q/K -> RoPE Q/K -> attention-ready Q/K
GQA packed QKV -> per-head RMSNorm Q/K -> rotate-half RoPE -> Q/K/V workspaces
packed QKV + bias -> split Q/K/V -> rotate-half RoPE Q/K
decode Q/K/V -> RMSNorm Q/K -> rotate-half RoPE Q/K -> Q stage / KV cache write
GQA packed QKV -> split Q/K/V -> interleaved RoPE Q/K -> KV cache write
```

## Exported APIs

- `qkv_split_bf16(packed_qkv, heads, head_dim, q_out=None, k_out=None, v_out=None)`
- `qkv_split_norm_rope_bf16(packed_qkv, norm_q_weight, norm_k_weight, freqs_re, freqs_im, heads, head_dim, rope_seq_len=None, eps=1e-6, q_out=None, k_out=None)`
- `qkv_split_bias_norm_rope_v_bf16(packed_qkv, qkv_bias, norm_q_weight, norm_k_weight, freqs_re, freqs_im, heads, head_dim, rope_seq_len=None, eps=1e-6, q_out=None, k_out=None, v_out=None)`
- `qkv_split_bias_norm_rope_v_cat_bf16(packed_qkv, qkv_bias, norm_q_weight, norm_k_weight, freqs_re, freqs_im, heads, head_dim, video_offset, q_cat_out, k_cat_out, v_cat_out, rope_seq_len=None, eps=1e-6)`
- `qkv_split_joint3_cat_bf16(packed_v, qkv_v_bias, norm_v_q_weight, norm_v_k_weight, freqs_re, freqs_im, packed_a, norm_a_q_weight, norm_a_k_weight, packed_u, norm_u_q_weight, norm_u_k_weight, heads, head_dim, q_cat_out, k_cat_out, v_cat_out, rope_seq_len=None, eps_v=1e-6, eps_a=1e-6, eps_u=1e-6)`
- `qkv_split_per_head_norm_rope_bf16(packed_qkv, q_norm_weight, k_norm_weight, cos, sin, q_heads, kv_heads, eps=1e-6, q_out=None, k_out=None, v_out=None)`
- `qkv_split_bias_rope_bf16(packed_qkv, qkv_bias, cos, sin, q_heads, kv_heads, head_dim, q_out=None, k_out=None, v_out=None)`
- `qkv_split_bias_rope_fp16(packed_qkv, qkv_bias, cos, sin, q_heads, kv_heads, head_dim, q_out=None, k_out=None, v_out=None)`
- `qkv_split_rope_kvcache_bf16(packed_qkv, rope, q_heads, kv_heads, head_dim, cache_offset, q_out=None, k_cache=None, v_cache=None, max_seq_len=None)`
- `qkv_split_rope_kvcache_fp16(packed_qkv, rope, q_heads, kv_heads, head_dim, cache_offset=0, device_position=None, q_out=None, k_cache=None, v_cache=None, max_seq_len=None)`

- `qk_norm_rope_strided_bf16(q, k, q_weight, k_weight, cos, sin, q_heads, k_heads, head_dim, eps=1e-6, q_out=None, k_out=None)`
- `decode_q_norm_rope_stage_bf16(q_pre, q_norm_weight, cos, sin, eps=1e-6, q_out=None)`
- `decode_k_norm_rope_kvwrite_bf16(k_pre, v_pre, k_norm_weight, cos, sin, eps=1e-6, k_cache_dst=None, v_cache_dst=None)`
- `decode_k_norm_rope_kvwrite_devpos_bf16(k_pre, v_pre, k_norm_weight, cos, sin, cur_pos, k_cache, v_cache, eps=1e-6)`

The BF16 and FP16 CUDA cache writers automatically use a 16-byte vectorized
path for aligned head dimensions and buffers, including the PI0.5 Thor
`B=1, S=50, QH=8, KVH=1, HD=256` shape. The ROCm BF16 cache writer provides
the same aligned/scalar dispatch for `gfx950`. Unaligned tensors retain the
scalar fallback; the public API and output contract are unchanged.

Tensor conventions:

- `packed_qkv`: BF16, shape `(batch, seq_len, 3 * heads * head_dim)`
- `norm_q_weight`: BF16, shape `(heads * head_dim,)`
- `norm_k_weight`: BF16, shape `(heads * head_dim,)`
- `freqs_re`, `freqs_im`: FP32, shape `(>= rope_seq_len, head_dim / 2)`
- non-cat outputs: BF16, shape `(batch, seq_len, heads, head_dim)`
- cat outputs: BF16, shape `(batch, total_seq_len, heads, head_dim)`
- GQA cache input `packed_qkv`: BF16, shape
  `(batch, seq_len, (q_heads + 2 * kv_heads) * head_dim)`
- per-head GQA `q_norm_weight` and `k_norm_weight`: BF16, shape `(128,)`
- per-head GQA `cos` and `sin`: BF16, shape `(batch, seq_len, 128)`,
  using the full rotate-half table contract
- per-head GQA outputs: BF16 Q shaped `(batch, seq_len, q_heads, 128)`
  and K/V shaped `(batch, seq_len, kv_heads, 128)`
- bias+RoPE `cos`/`sin`: FP32, shape
  `(batch, seq_len, head_dim / 2)` or `(batch, seq_len, head_dim)`
- bias+RoPE supports even `head_dim` values in `[8, 256]` and unequal Q/KV
  head counts
- the FP16-output bias+RoPE variant keeps BF16 projected QKV and bias inputs,
  but writes FP16 Q/K/V directly for an FP16 attention island
- GQA `rope`: BF16, shape `(>= seq_len, head_dim)`, interleaved
  `[cos0, sin0, cos1, sin1, ...]` per row
- GQA cache `q_out`: BF16, shape `(batch, seq_len, q_heads, head_dim)`
- GQA `k_cache`/`v_cache`: BF16, shape
  `(batch, max_seq_len, kv_heads, head_dim)`; writes start at host integer
  `cache_offset`
- decode `q_pre`, `k_pre`, `v_pre`: BF16, shape `(heads, 128)`
- decode `cos`, `sin`: BF16, shape `(64,)`, rotate-half RoPE contract
- decode cache direct outputs: BF16, shape `(n_kv_heads, 128)`
- decode devpos cache tensors: BF16, shape `(max_seq_len, n_kv_heads, 128)`;
  `cur_pos` is a CUDA int32 scalar or one-element tensor.

The package includes both standalone Q/K(/V) output APIs and direct joint
workspace APIs. The `joint3` API is the VLA-oriented path: it fuses video QKV
bias + Q/K RMSNorm + RoPE, action Q/K RMSNorm, und Q/K RMSNorm, and direct
Q/K/V writes into a single attention-ready workspace.

The decode APIs are single-token hot-path helpers. `decode_q_norm_rope_stage`
writes Q staging buffers. The KV-write APIs write K/V directly into either a
caller-provided cache slot or a slot selected by a device-side `cur_pos`, which
is useful for CUDA Graph replay.

The GQA cache API is the sequence form used by static decoder loops. It does
not apply RMSNorm; callers should pass already-projected QKV rows. It applies
the same adjacent-pair, interleaved cos/sin RoPE contract used by FlashRT
PI0/PI0.5 RTX decoder staging, but the public API remains model-agnostic.
The FP16 twin accepts either a host `cache_offset` or a CUDA int32
`device_position`; the latter keeps cache position inside a captured graph.
`qk_norm_rope_strided_bf16` handles independently strided Q/K projections and
is validated on the Cosmos3-Edge `(rows=51, Hq=16, Hk=2, D=128)` contract.

The per-head GQA API covers sequence attention blocks that normalize every
128-wide Q/K head independently before rotate-half RoPE. It accepts unequal
Q/KV head counts and writes contiguous Q/K/V workspaces, so callers can feed
the result directly to an attention core without materializing separate split,
normalization, and RoPE intermediates.

The bias+RoPE API is the generic vision/text projection epilogue. It consumes
one packed GQA or equal-head QKV GEMM output, adds the broadcast bias, applies
split-half RoPE to Q/K, and writes attention-ready Q/K/V in one launch.

## Backend support

- CUDA 12.8+: all APIs listed above.
- ROCm 7.2 on `gfx950`: `qkv_split_rope_kvcache_bf16`.

The package metadata is the hardware source of truth. A ROCm build does not
register CUDA-only implementations, so an unsupported symbol fails explicitly
instead of silently taking an eager replacement path.

## Minimal Usage

```python
from kernels import get_kernel
import torch

ops = get_kernel("flashrt/flashrt-qkv-cache-rope", version=1, trust_remote_code=True)

packed_qkv = torch.randn((1, 1024, 3 * 24 * 128), device="cuda", dtype=torch.bfloat16)
q_w = torch.ones((24 * 128,), device="cuda", dtype=torch.bfloat16)
k_w = torch.ones((24 * 128,), device="cuda", dtype=torch.bfloat16)
theta = torch.randn((1024, 64), device="cuda")
freqs_re = torch.cos(theta).contiguous()
freqs_im = torch.sin(theta).contiguous()

q, k = ops.qkv_split_norm_rope_bf16(
    packed_qkv,
    q_w,
    k_w,
    freqs_re,
    freqs_im,
    heads=24,
    head_dim=128,
)
```

VLA joint workspace path:

```python
q_cat, k_cat, v_cat = ops.qkv_split_joint3_cat_bf16(
    packed_v,
    qkv_v_bias,
    norm_v_q_weight,
    norm_v_k_weight,
    freqs_re,
    freqs_im,
    packed_a,
    norm_a_q_weight,
    norm_a_k_weight,
    packed_u,
    norm_u_q_weight,
    norm_u_k_weight,
    heads=24,
    head_dim=128,
    q_cat_out=q_cat,
    k_cat_out=k_cat,
    v_cat_out=v_cat,
)
```

Decode cache-write path:

```python
q_pre = torch.randn((24, 128), device="cuda", dtype=torch.bfloat16)
k_pre = torch.randn((8, 128), device="cuda", dtype=torch.bfloat16)
v_pre = torch.randn((8, 128), device="cuda", dtype=torch.bfloat16)
q_w = torch.ones((128,), device="cuda", dtype=torch.bfloat16)
k_w = torch.ones((128,), device="cuda", dtype=torch.bfloat16)
cos = torch.ones((64,), device="cuda", dtype=torch.bfloat16)
sin = torch.zeros((64,), device="cuda", dtype=torch.bfloat16)

q_buf = ops.decode_q_norm_rope_stage_bf16(q_pre, q_w, cos, sin)

k_cache = torch.empty((4096, 8, 128), device="cuda", dtype=torch.bfloat16)
v_cache = torch.empty_like(k_cache)
cur_pos = torch.tensor([17], device="cuda", dtype=torch.int32)
ops.decode_k_norm_rope_kvwrite_devpos_bf16(
    k_pre,
    v_pre,
    k_w,
    cos,
    sin,
    cur_pos,
    k_cache,
    v_cache,
)
```

GQA sequence cache-write path:

```python
packed_qkv = torch.randn((1, 10, (8 + 2 * 1) * 256), device="cuda", dtype=torch.bfloat16)
rope = torch.empty((10, 256), device="cuda", dtype=torch.bfloat16)  # [cos,sin] pairs
q = torch.empty((1, 10, 8, 256), device="cuda", dtype=torch.bfloat16)
k_cache = torch.empty((1, 560 + 10, 1, 256), device="cuda", dtype=torch.bfloat16)
v_cache = torch.empty_like(k_cache)

ops.qkv_split_rope_kvcache_bf16(
    packed_qkv,
    rope,
    q_heads=8,
    kv_heads=1,
    head_dim=256,
    cache_offset=560,
    q_out=q,
    k_cache=k_cache,
    v_cache=v_cache,
)
```

Per-head GQA sequence path:

```python
packed_qkv = torch.randn(
    (1, 277, (16 + 2 * 8) * 128),
    device="cuda",
    dtype=torch.bfloat16,
)
q_w = torch.ones((128,), device="cuda", dtype=torch.bfloat16)
k_w = torch.ones((128,), device="cuda", dtype=torch.bfloat16)
cos = torch.ones((1, 277, 128), device="cuda", dtype=torch.bfloat16)
sin = torch.zeros_like(cos)

q, k, v = ops.qkv_split_per_head_norm_rope_bf16(
    packed_qkv,
    q_w,
    k_w,
    cos,
    sin,
    q_heads=16,
    kv_heads=8,
)
```

## Validation

```bash
python flashrt-qkv-cache-rope/tests/test_qkv_cache_rope.py --backend source --mode full
python flashrt-qkv-cache-rope/benchmarks/benchmark.py --backend source --shapes all
```

Current RTX 5090 source-extension rows pass across no-bias, bias+V,
bias+cat-workspace, joint3 workspace, decode Q stage, direct KV-write, and
device-position KV-write paths. The source test grid also covers the
GQA sequence KV-cache path used by static decoder loops. The source benchmark
and correctness grid also cover the per-head GQA sequence boundary at
`B=1, S=277, QH=16, KVH=8, HD=128`. The source benchmark includes
single-stream Wan/VL rows, VLA-shaped `joint3_vla`, and single-token decode
rows. Current source rows show roughly `24x-38x` for single-stream QKV
postprocess, `36x-86x` for joint workspace rows, and `29x-30x` for decode
Q/KV-write rows against PyTorch eager references on RTX 5090. Built-artifact
and multi-hardware validation are pending for this expanded API surface.
