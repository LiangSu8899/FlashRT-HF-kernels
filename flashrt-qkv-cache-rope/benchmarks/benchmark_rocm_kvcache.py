#!/usr/bin/env python3
"""Paired ROCm benchmark for the Pi0.5 packed-QKV/RoPE/cache seam.

The comparison boundary is identical on both arms: split packed GQA QKV,
apply adjacent-pair RoPE to Q/K, and write Q plus dense K/V cache slots.
The baseline is a full-graph ``torch.compile(mode="max-autotune")`` chain.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
COMMON_PATH = HERE / "benchmark.py"


def _load_common():
    spec = importlib.util.spec_from_file_location(
        "flashrt_qkv_cache_rope_benchmark_common", COMMON_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark helpers from {COMMON_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _paired_device_times(kernel_fn, baseline_fn, warmup: int, iters: int):
    for _ in range(warmup):
        kernel_fn()
        baseline_fn()
    torch.cuda.synchronize()

    records = []
    for iteration in range(iters):
        order = (
            (("kernel", kernel_fn), ("baseline", baseline_fn))
            if iteration % 2 == 0
            else (("baseline", baseline_fn), ("kernel", kernel_fn))
        )
        row = {}
        for name, fn in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            row[name] = (start, end)
        records.append(row)
    torch.cuda.synchronize()

    kernel_us = [row["kernel"][0].elapsed_time(row["kernel"][1]) * 1000 for row in records]
    baseline_us = [
        row["baseline"][0].elapsed_time(row["baseline"][1]) * 1000
        for row in records
    ]
    paired_speedups = [b / k for k, b in zip(kernel_us, baseline_us)]
    return kernel_us, baseline_us, paired_speedups


def _make_case(common, batch, seq_len, q_heads, kv_heads, head_dim):
    width = (q_heads + 2 * kv_heads) * head_dim
    packed = torch.randn(
        (batch, seq_len, width), device="cuda", dtype=torch.bfloat16
    )
    rope = common.make_interleaved_rope(seq_len, head_dim)
    cache_offset = 2
    max_seq_len = cache_offset + seq_len + 2
    q_out = torch.empty(
        (batch, seq_len, q_heads, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    k_cache = torch.empty(
        (batch, max_seq_len, kv_heads, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v_cache = torch.empty_like(k_cache)
    return packed, rope, cache_offset, q_out, k_cache, v_cache


def _benchmark_shape(common, ops, shape, args):
    name, batch, seq_len, q_heads, kv_heads, head_dim = shape
    packed, rope, offset, q_out, k_cache, v_cache = _make_case(
        common, batch, seq_len, q_heads, kv_heads, head_dim
    )
    target = slice(offset, offset + seq_len)

    def kernel_chain(packed_arg, rope_arg, q_arg, k_arg, v_arg):
        return ops.qkv_split_rope_kvcache_bf16(
            packed_arg,
            rope_arg,
            q_heads,
            kv_heads,
            head_dim,
            offset,
            q_arg,
            k_arg,
            v_arg,
        )

    def baseline_chain(packed_arg, rope_arg, q_arg, k_arg, v_arg):
        q, k, v = common.torch_ref_kvcache(
            packed_arg, rope_arg, q_heads, kv_heads, head_dim
        )
        q_arg.copy_(q)
        k_arg[:, target].copy_(k)
        v_arg[:, target].copy_(v)
        return q_arg, k_arg, v_arg

    compiled_kernel = torch.compile(
        kernel_chain, fullgraph=True, mode=args.compile_mode
    )
    compiled_baseline = torch.compile(
        baseline_chain, fullgraph=True, mode=args.compile_mode
    )

    def kernel_fn():
        return compiled_kernel(packed, rope, q_out, k_cache, v_cache)

    def baseline_fn():
        return compiled_baseline(packed, rope, q_out, k_cache, v_cache)

    kernel_fn()
    torch.cuda.synchronize()
    got = (q_out.clone(), k_cache[:, target].clone(), v_cache[:, target].clone())
    expected = common.torch_ref_kvcache(
        packed, rope, q_heads, kv_heads, head_dim
    )
    metrics = [common.metrics(actual, reference) for actual, reference in zip(got, expected)]
    worst_p99 = max(item[0] for item in metrics)
    worst_cosine = min(item[1] for item in metrics)

    kernel_us, baseline_us, speedups = _paired_device_times(
        kernel_fn, baseline_fn, args.warmup, args.iters
    )
    result = {
        "shape": name,
        "batch": batch,
        "seq_len": seq_len,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "dtype": "bfloat16",
        "compile_mode": args.compile_mode,
        "comparison_boundary": "split + adjacent-pair RoPE + Q/K/V cache writes",
        "kernel_us_median": statistics.median(kernel_us),
        "kernel_us_p10": _percentile(kernel_us, 0.10),
        "kernel_us_p90": _percentile(kernel_us, 0.90),
        "compiled_baseline_us_median": statistics.median(baseline_us),
        "compiled_baseline_us_p10": _percentile(baseline_us, 0.10),
        "compiled_baseline_us_p90": _percentile(baseline_us, 0.90),
        "paired_speedup_median": statistics.median(speedups),
        "paired_speedup_p10": _percentile(speedups, 0.10),
        "paired_speedup_p90": _percentile(speedups, 0.90),
        "worst_p99_abs": worst_p99,
        "worst_cosine": worst_cosine,
        "status": (
            "PASS"
            if worst_p99 <= args.p99_abs_limit
            and worst_cosine >= args.cosine_limit
            else "FAIL"
        ),
    }
    print(
        f"{result['status']} {name}: kernel={result['kernel_us_median']:.3f}us "
        f"compiled={result['compiled_baseline_us_median']:.3f}us "
        f"paired={result['paired_speedup_median']:.3f}x "
        f"p99={worst_p99:.6f} cosine={worst_cosine:.9f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact")
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--p99-abs-limit", type=float, default=0.015625)
    parser.add_argument("--cosine-limit", type=float, default=0.999)
    parser.add_argument("--output")
    args = parser.parse_args()

    if torch.version.hip is None or not torch.cuda.is_available():
        raise SystemExit("an available ROCm device is required")
    torch.manual_seed(41)
    common = _load_common()
    ops = (
        common.load_source_ops()
        if args.backend == "source"
        else common.load_installed_ops(args.artifact)
    )
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    shapes = [
        ("pi05_decoder", 1, 10, 8, 1, 256),
        ("pi05_prefix", 1, 712, 8, 1, 256),
        ("cross_family_batch2", 2, 16, 8, 2, 128),
    ]
    results = [_benchmark_shape(common, ops, shape, args) for shape in shapes]
    report = {
        "torch": torch.__version__,
        "rocm": torch.version.hip,
        "device": device.name,
        "arch": device.gcnArchName,
        "timing": "paired alternating CUDA/HIP events",
        "results": results,
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    if any(result["status"] != "PASS" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
