# Validation: flashrt-qkv-cache-rope

Required before publishing this package:

1. Source-extension correctness:

   ```bash
   python flashrt-qkv-cache-rope/tests/test_qkv_cache_rope.py --backend source --mode full
   ```

   The full mode includes the GQA sequence cache-write API
   `qkv_split_rope_kvcache_bf16` with a PI0.5 decoder-shaped row
   `(B=1, S=10, q_heads=8, kv_heads=1, head_dim=256)`, a batched GQA row,
   cache prefix/suffix preservation checks, and invalid shape/bounds rejection.
   It also covers the per-head GQA norm/RoPE/staging API with a small row and
   an N1.7-shaped row `(B=1, S=277, q_heads=16, kv_heads=8, head_dim=128)`.
   The fused bias+RoPE matrix covers GROOT vision, Qwen3-VL vision/text,
   LingBot vision, and both half-width and full-width RoPE tables.
   It also covers FP16 output for a GQA HD80 attention-island shape.

2. Source-extension benchmark:

   ```bash
   python flashrt-qkv-cache-rope/benchmarks/benchmark.py \
     --backend source \
     --shapes all \
     --warmup 3 \
     --iters 10
   ```

   Fused bias+RoPE eager/compile comparison:

   ```bash
   python flashrt-qkv-cache-rope/benchmarks/benchmark_bias_rope.py \
     --backend source
   ```

3. Kernel-builder artifact build:

   ```bash
   kernel-builder build-and-copy flashrt-qkv-cache-rope
   ```

4. Built-artifact correctness:

   ```bash
   PYTHONPATH=<artifact-path> \
   python flashrt-qkv-cache-rope/tests/test_qkv_cache_rope.py \
     --backend installed \
     --mode full
   ```

5. Built-artifact benchmark:

   ```bash
   python flashrt-qkv-cache-rope/benchmarks/benchmark.py \
     --backend installed \
     --artifact <artifact-path> \
     --shapes all
   ```

6. Multi-hardware matrix:

   Add hardware claims only after the same correctness and benchmark commands
   pass on that machine.

### ROCm `gfx950` gate

The ROCm artifact currently contains only `qkv_split_rope_kvcache_bf16`.
Validate it at the Pi0.5 decoder and prefix rows plus an unrelated batched-GQA
row:

```bash
PYTORCH_ROCM_ARCH=gfx950 \
python flashrt-qkv-cache-rope/tests/test_qkv_cache_rope.py \
  --backend source --mode full

PYTORCH_ROCM_ARCH=gfx950 \
python flashrt-qkv-cache-rope/benchmarks/benchmark_rocm_kvcache.py \
  --backend source --compile-mode max-autotune
```

The benchmark is paired and alternating. Its baseline is the identical
split/RoPE/cache-write boundary compiled with
`torch.compile(fullgraph=True, mode="max-autotune")`; an eager-only comparison
is not a promotion gate. Repeat both commands with `--backend installed` and
the exact `torch211-cxx11-rocm72-x86_64-linux` artifact before publishing the
hardware claim.

## Local validation log

2026-08-24 (ROCm owner preflight, commit `4fa73b2`):

- The repository static prebuild check and the exact pinned-builder
  `check-config` gate passed for
  `torch211-cxx11-rocm72-x86_64-linux`. Fork CI run `32752196310` then stopped
  at the expected empty fork `HF_TOKEN`; it did not publish an artifact.
- The source extension compiled cleanly for `gfx950` with PyTorch 2.11 / ROCm
  7.2 and the registration header from the package-pinned kernel-builder
  commit `432702bfbfbb17d3a1bd2c2743d004e21e769ab7`. This was a cross-compile
  gate on a non-AMD worker, not launch or correctness evidence. MI350X
  correctness, compile/capture, paired performance, and installed-artifact
  gates remain pending.

2026-08-06:

- The expanded source gate passed on NVIDIA Thor (SM110), PyTorch
  `2.13.0+cu130`, and CUDA 13.0.
- `qkv_split_rope_kvcache_fp16` and its CUDA-device-position form passed the
  PI0.5 D256 static-cache rows. Worst p99 absolute error was `0.005859`, worst
  cosine was `0.99999857`, V-cache writes were exact, and graph replay was
  bitwise deterministic.
- `qk_norm_rope_strided_bf16` passed the Cosmos Edge row with Q/K p99 absolute
  error `0.015625`, cosine at least `0.99999559`, and bitwise graph replay.

2026-07-30:

- `qkv_split_bias_rope_bf16` passed full source correctness for head dimensions
  64, 72, 80, and 128. The native migration harness compared the Tensor
  wrapper directly with FlashRT's original pointer API on five model-shaped
  rows: all Q/K/V outputs were bitwise exact and wrapper/native latency ratios
  ranged from `1.000` to `1.001`.
- On RTX 5090, the fused bias+RoPE wrapper measured `1.42x` to `6.28x` faster
  than the equivalent `torch.compile(fullgraph=True)` chain and `3.74x` to
  `11.34x` faster than eager across the published matrix.
- Source correctness passed for the expanded API, including the N1.7-shaped
  per-head GQA row. Q/K p99 absolute error was `0.015625`, cosine was at least
  `0.99999619`, and V staging was exact.
- At `(B=1, S=277, q_heads=16, kv_heads=8, head_dim=128)`, the fused
  per-head GQA boundary measured `6.154 us` versus `35.798 us` for a
  `torch.compile(fullgraph=True)` split, per-head norm, rotate-half RoPE, and
  Q/K/V staging chain: `5.82x` faster on RTX 5090.
- An installed-wrapper-equivalent source build passed
  `torch.compile(fullgraph=True)` through the public Python API, including
  fake registration and the default-epsilon call contract. The installed
  test suite now carries this regression for release artifacts.
- `kernel-builder-docker check-config .` passed for the repository. Artifact
  rebuild was blocked by external Nix/Cachix fetch timeouts; source compilation
  and execution passed.

2026-06-09:

- Source correctness passed:

  ```bash
  python flashrt-qkv-cache-rope/tests/test_qkv_cache_rope.py \
    --backend source \
    --mode full
  ```

- Source smoke benchmark passed in `pi0-stablehlo-test` with the isolated
  `internal-tests/envs/openpi-baseline` venv. The PI0.5 GQA cache row reported
  exact Q/K/V agreement and `4.998 us` FlashRT time.

- Kernel-builder config, build, build check, and ABI check passed:

  ```bash
  kernel-builder-docker check-config .
  kernel-builder-docker build --variant torch212-cxx11-cu132-x86_64-linux .
  kernel-builder-docker build --variant torch211-cxx11-cu128-x86_64-linux .
  kernel-builder-docker check-builds .
  kernel-builder-docker check-abi .
  ```

- Built-artifact correctness passed with
  `build/torch211-cxx11-cu128-x86_64-linux` in the matching
  `.flashrt-hub-smoke-torch211` venv:

  ```bash
  PYTHONPATH=flashrt-qkv-cache-rope/build/torch211-cxx11-cu128-x86_64-linux \
  python flashrt-qkv-cache-rope/tests/test_qkv_cache_rope.py \
    --backend installed \
    --mode full
  ```

- Built-artifact benchmark passed with
  `build/torch211-cxx11-cu128-x86_64-linux`; results are recorded in
  `benchmarks/RESULTS.md`. The PI0.5 GQA cache row reported exact Q/K
  agreement for `(B=1, S=10, q_heads=8, kv_heads=1, head_dim=256)` and
  `3.802 us` FlashRT time.

- `build/torch212-cxx11-cu132-x86_64-linux` imports fail in the
  `pi0-stablehlo-test` Torch 2.9 nightly venv with a Torch C++ ABI undefined
  symbol. This is an environment/variant mismatch; `check-abi` passes for the
  artifact, and the torch211/cu128 artifact passes in its matching venv.
