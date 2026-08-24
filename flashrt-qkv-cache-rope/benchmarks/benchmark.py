#!/usr/bin/env python3
"""Benchmark flashrt-qkv-cache-rope against a PyTorch eager postprocess chain."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "flashrt-qkv-cache-rope"
_DEFAULT_REGISTRATION_INCLUDE = (
    ROOT.parent
    / "kernels"
    / "kernel-builder"
    / "src"
    / "pyproject"
    / "templates"
    / "torch"
)
REGISTRATION_INCLUDE = Path(
    os.environ.get(
        "KERNEL_BUILDER_TORCH_INCLUDE", str(_DEFAULT_REGISTRATION_INCLUDE)
    )
)

SHAPES = {
    "small": (1, 64, 8, 128),
    "wan_1k": (1, 1024, 24, 128),
    "wan_2520": (1, 2520, 24, 128),
    "wan_4096": (1, 4096, 24, 128),
    "vl_512": (1, 512, 16, 128),
}
SHAPE_GROUPS = {
    "smoke": ["small"],
    "headline": ["wan_1k", "wan_2520", "vl_512"],
    "all": list(SHAPES.keys()),
}


@dataclass
class Result:
    shape: str
    batch: int
    seq_len: int
    heads: int
    head_dim: int
    flashrt_us: float
    torch_eager_us: float
    speedup_vs_eager: float
    q_p99_abs: float
    k_p99_abs: float
    q_cosine: float
    k_cosine: float
    status: str


class SourceOps:
    def __init__(self, namespace: str) -> None:
        self._ops = getattr(torch.ops, namespace)

    def decode_q_norm_rope_stage_bf16(self, q_pre, q_w, cos, sin, eps=1e-6, q_out=None):
        if q_out is None:
            q_out = torch.empty_like(q_pre)
        self._ops.decode_q_norm_rope_stage_bf16(q_pre, q_w, cos, sin, float(eps), q_out)
        return q_out

    def decode_k_norm_rope_kvwrite_bf16(self, k_pre, v_pre, k_w, cos, sin, eps=1e-6, k_out=None, v_out=None):
        if k_out is None:
            k_out = torch.empty_like(k_pre)
        if v_out is None:
            v_out = torch.empty_like(v_pre)
        self._ops.decode_k_norm_rope_kvwrite_bf16(k_pre, v_pre, k_w, cos, sin, float(eps), k_out, v_out)
        return k_out, v_out

    def decode_k_norm_rope_kvwrite_devpos_bf16(self, k_pre, v_pre, k_w, cos, sin, cur_pos, k_cache, v_cache, eps=1e-6):
        self._ops.decode_k_norm_rope_kvwrite_devpos_bf16(k_pre, v_pre, k_w, cos, sin, cur_pos, float(eps), k_cache, v_cache)
        return k_cache, v_cache

    def qkv_split_norm_rope_bf16(
        self, packed, q_w, k_w, freqs_re, freqs_im, heads, head_dim, rope_seq_len=None, eps=1e-6, q_out=None, k_out=None
    ):
        if rope_seq_len is None:
            rope_seq_len = packed.shape[1]
        if q_out is None:
            q_out = torch.empty((packed.shape[0], packed.shape[1], heads, head_dim), device=packed.device, dtype=torch.bfloat16)
        if k_out is None:
            k_out = torch.empty_like(q_out)
        self._ops.qkv_split_norm_rope_bf16(
            packed, q_w, k_w, freqs_re, freqs_im, int(heads), int(head_dim),
            int(rope_seq_len), float(eps), q_out, k_out
        )
        return q_out, k_out

    def qkv_split_bias_rope_bf16(
        self,
        packed,
        qkv_bias,
        cos,
        sin,
        q_heads,
        kv_heads,
        head_dim,
        q_out=None,
        k_out=None,
        v_out=None,
    ):
        batch, seq_len, _ = packed.shape
        if q_out is None:
            q_out = torch.empty(
                (batch, seq_len, q_heads, head_dim),
                device=packed.device,
                dtype=torch.bfloat16,
            )
        if k_out is None:
            k_out = torch.empty(
                (batch, seq_len, kv_heads, head_dim),
                device=packed.device,
                dtype=torch.bfloat16,
            )
        if v_out is None:
            v_out = torch.empty_like(k_out)
        self._ops.qkv_split_bias_rope_bf16(
            packed,
            qkv_bias,
            cos,
            sin,
            int(q_heads),
            int(kv_heads),
            int(head_dim),
            q_out,
            k_out,
            v_out,
        )
        return q_out, k_out, v_out

    def qkv_split_bias_rope_fp16(
        self,
        packed,
        qkv_bias,
        cos,
        sin,
        q_heads,
        kv_heads,
        head_dim,
        q_out=None,
        k_out=None,
        v_out=None,
    ):
        batch, seq_len, _ = packed.shape
        if q_out is None:
            q_out = torch.empty(
                (batch, seq_len, q_heads, head_dim),
                device=packed.device,
                dtype=torch.float16,
            )
        if k_out is None:
            k_out = torch.empty(
                (batch, seq_len, kv_heads, head_dim),
                device=packed.device,
                dtype=torch.float16,
            )
        if v_out is None:
            v_out = torch.empty_like(k_out)
        self._ops.qkv_split_bias_rope_fp16(
            packed,
            qkv_bias,
            cos,
            sin,
            int(q_heads),
            int(kv_heads),
            int(head_dim),
            q_out,
            k_out,
            v_out,
        )
        return q_out, k_out, v_out

    def qkv_split_joint3_cat_bf16(
        self,
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
        heads,
        head_dim,
        q_cat_out,
        k_cat_out,
        v_cat_out,
        rope_seq_len=None,
        eps_v=1e-6,
        eps_a=1e-6,
        eps_u=1e-6,
    ):
        if rope_seq_len is None:
            rope_seq_len = packed_v.shape[1]
        self._ops.qkv_split_joint3_cat_bf16(
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
            int(heads),
            int(head_dim),
            int(rope_seq_len),
            float(eps_v),
            float(eps_a),
            float(eps_u),
            q_cat_out,
            k_cat_out,
            v_cat_out,
        )
        return q_cat_out, k_cat_out, v_cat_out

    def qkv_split_rope_kvcache_bf16(
        self,
        packed_qkv,
        rope,
        q_heads,
        kv_heads,
        head_dim,
        cache_offset,
        q_out=None,
        k_cache=None,
        v_cache=None,
        max_seq_len=None,
    ):
        batch, seq_len, _ = packed_qkv.shape
        if q_out is None:
            q_out = torch.empty(
                (batch, seq_len, q_heads, head_dim),
                device=packed_qkv.device,
                dtype=torch.bfloat16,
            )
        if k_cache is None or v_cache is None:
            if max_seq_len is None:
                max_seq_len = cache_offset + seq_len
            cache_shape = (batch, max_seq_len, kv_heads, head_dim)
            if k_cache is None:
                k_cache = torch.empty(cache_shape, device=packed_qkv.device, dtype=torch.bfloat16)
            if v_cache is None:
                v_cache = torch.empty(cache_shape, device=packed_qkv.device, dtype=torch.bfloat16)
        self._ops.qkv_split_rope_kvcache_bf16(
            packed_qkv,
            rope,
            int(q_heads),
            int(kv_heads),
            int(head_dim),
            int(cache_offset),
            q_out,
            k_cache,
            v_cache,
        )
        return q_out, k_cache, v_cache


def _preload_cublaslt() -> None:
    for parent in Path(torch.__file__).resolve().parents:
        candidate = parent / "nvidia" / "cublas" / "lib" / "libcublasLt.so.12"
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return
    library = ctypes.util.find_library("cublasLt")
    if library:
        ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)


def _current_arch_list() -> str:
    major, minor = torch.cuda.get_device_capability(0)
    return f"{major}.{minor}"


def load_source_ops() -> SourceOps:
    from torch.utils.cpp_extension import load

    if not REGISTRATION_INCLUDE.is_dir():
        raise RuntimeError(f"missing kernel-builder registration include: {REGISTRATION_INCLUDE}")
    _preload_cublaslt()
    is_rocm = torch.version.hip is not None
    if is_rocm:
        arch = os.environ.get("PYTORCH_ROCM_ARCH")
        if not arch:
            arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
        os.environ.setdefault("PYTORCH_ROCM_ARCH", arch)
        sources = [
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "rocm" / "qkv_cache_rope_rocm.hip"),
        ]
        cflags = ["-O3", "-DROCM_KERNEL"]
        device_cflags = ["-O3", "-DROCM_KERNEL"]
    else:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _current_arch_list())
        sources = [
            str(PACKAGE / "torch-ext" / "torch_binding.cpp"),
            str(PACKAGE / "csrc" / "qkv_cache_rope.cu"),
        ]
        cflags = ["-O3", "-DCUDA_KERNEL"]
        device_cflags = [
            "-O3", "--expt-relaxed-constexpr", "-DCUDA_KERNEL"
        ]
    namespace = "flashrt_qkv_cache_rope_benchmark"
    load(
        name=namespace,
        sources=sources,
        extra_include_paths=[str(PACKAGE / "csrc"), str(REGISTRATION_INCLUDE)],
        extra_cflags=cflags,
        extra_cuda_cflags=device_cflags,
        verbose=False,
    )
    return SourceOps(namespace)


def load_installed_ops(artifact: str | None):
    if artifact:
        sys.path.insert(0, artifact)
    try:
        return importlib.import_module("flashrt_qkv_cache_rope")
    finally:
        if artifact:
            sys.path.remove(artifact)


def make_freqs(seq_len: int, head_dim: int):
    theta = torch.randn((seq_len, head_dim // 2), device="cuda", dtype=torch.float32)
    return torch.cos(theta).contiguous(), torch.sin(theta).contiguous()


def make_interleaved_rope(seq_len: int, head_dim: int):
    theta = torch.randn((seq_len, head_dim // 2), device="cuda", dtype=torch.float32)
    cos = torch.cos(theta).to(torch.bfloat16)
    sin = torch.sin(theta).to(torch.bfloat16)
    return torch.stack([cos, sin], dim=-1).reshape(seq_len, head_dim).contiguous()


def make_case(batch: int, seq_len: int, heads: int, head_dim: int):
    dim = heads * head_dim
    packed = torch.randn((batch, seq_len, 3 * dim), device="cuda", dtype=torch.bfloat16)
    q_w = (1.0 + 0.1 * torch.randn((dim,), device="cuda", dtype=torch.bfloat16)).contiguous()
    k_w = (1.0 + 0.1 * torch.randn((dim,), device="cuda", dtype=torch.bfloat16)).contiguous()
    freqs_re, freqs_im = make_freqs(seq_len, head_dim)
    q_out = torch.empty((batch, seq_len, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k_out = torch.empty_like(q_out)
    return packed, q_w, k_w, freqs_re, freqs_im, q_out, k_out


def make_decode_case(heads: int):
    q = torch.randn((heads, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((heads, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((heads, 128), device="cuda", dtype=torch.bfloat16)
    q_w = (1.0 + 0.1 * torch.randn((128,), device="cuda", dtype=torch.bfloat16)).contiguous()
    k_w = (1.0 + 0.1 * torch.randn((128,), device="cuda", dtype=torch.bfloat16)).contiguous()
    theta = torch.randn((64,), device="cuda", dtype=torch.float32)
    cos = torch.cos(theta).to(torch.bfloat16).contiguous()
    sin = torch.sin(theta).to(torch.bfloat16).contiguous()
    return q, k, v, q_w, k_w, cos, sin


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    rms = torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps)
    return x.float() * rms * weight.float()


def apply_pair_rope(x: torch.Tensor, freqs_re: torch.Tensor, freqs_im: torch.Tensor):
    batch, seq_len, heads, head_dim = x.shape
    pair = x.float().reshape(batch, seq_len, heads, head_dim // 2, 2)
    re = pair[..., 0]
    im = pair[..., 1]
    fr = freqs_re.view(1, seq_len, 1, head_dim // 2)
    fi = freqs_im.view(1, seq_len, 1, head_dim // 2)
    out = torch.empty_like(pair.float())
    out[..., 0] = re * fr - im * fi
    out[..., 1] = re * fi + im * fr
    return out.reshape(batch, seq_len, heads, head_dim).to(torch.bfloat16)


def apply_interleaved_pair_rope(x: torch.Tensor, rope: torch.Tensor):
    batch, seq_len, heads, head_dim = x.shape
    pair = x.float().reshape(batch, seq_len, heads, head_dim // 2, 2)
    re = pair[..., 0]
    im = pair[..., 1]
    rope_pair = rope[:seq_len].float().reshape(seq_len, head_dim // 2, 2)
    cos = rope_pair[..., 0].view(1, seq_len, 1, head_dim // 2)
    sin = rope_pair[..., 1].view(1, seq_len, 1, head_dim // 2)
    out = torch.empty_like(pair.float())
    out[..., 0] = re * cos - im * sin
    out[..., 1] = re * sin + im * cos
    return out.reshape(batch, seq_len, heads, head_dim).to(torch.bfloat16)


def apply_rotate_half_rope_128(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    xf = x.float()
    out = torch.empty_like(xf)
    c = cos.float().view(1, 64)
    s = sin.float().view(1, 64)
    out[:, :64] = xf[:, :64] * c - xf[:, 64:] * s
    out[:, 64:] = xf[:, 64:] * c + xf[:, :64] * s
    return out.to(torch.bfloat16)


def torch_ref(packed, q_w, k_w, freqs_re, freqs_im, heads, head_dim, eps):
    batch, seq_len, _ = packed.shape
    dim = heads * head_dim
    q = packed[:, :, :dim]
    k = packed[:, :, dim : 2 * dim]
    qn = rms_norm(q, q_w, eps).to(torch.bfloat16).view(batch, seq_len, heads, head_dim)
    kn = rms_norm(k, k_w, eps).to(torch.bfloat16).view(batch, seq_len, heads, head_dim)
    return apply_pair_rope(qn, freqs_re, freqs_im), apply_pair_rope(kn, freqs_re, freqs_im)


def torch_ref_bias(packed, qkv_bias, q_w, k_w, freqs_re, freqs_im, heads, head_dim, eps):
    batch, seq_len, _ = packed.shape
    dim = heads * head_dim
    biased = packed.float() + qkv_bias.float().view(1, 1, 3 * dim)
    q = biased[:, :, :dim]
    k = biased[:, :, dim : 2 * dim]
    v = biased[:, :, 2 * dim :].to(torch.bfloat16).view(batch, seq_len, heads, head_dim)
    qn = rms_norm(q, q_w, eps).to(torch.bfloat16).view(batch, seq_len, heads, head_dim)
    kn = rms_norm(k, k_w, eps).to(torch.bfloat16).view(batch, seq_len, heads, head_dim)
    return apply_pair_rope(qn, freqs_re, freqs_im), apply_pair_rope(kn, freqs_re, freqs_im), v


def torch_ref_no_rope(packed, q_w, k_w, heads, head_dim, eps):
    batch, seq_len, _ = packed.shape
    dim = heads * head_dim
    q = packed[:, :, :dim]
    k = packed[:, :, dim : 2 * dim]
    v = packed[:, :, 2 * dim :].view(batch, seq_len, heads, head_dim)
    qn = rms_norm(q, q_w, eps).to(torch.bfloat16).view(batch, seq_len, heads, head_dim)
    kn = rms_norm(k, k_w, eps).to(torch.bfloat16).view(batch, seq_len, heads, head_dim)
    return qn, kn, v


def torch_ref_decode(x, weight, cos, sin, eps):
    return apply_rotate_half_rope_128(rms_norm(x, weight, eps).to(torch.bfloat16), cos, sin)


def torch_ref_kvcache(packed_qkv, rope, q_heads, kv_heads, head_dim):
    batch, seq_len, _ = packed_qkv.shape
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    q = packed_qkv[:, :, :q_dim].view(batch, seq_len, q_heads, head_dim)
    k = packed_qkv[:, :, q_dim : q_dim + kv_dim].view(batch, seq_len, kv_heads, head_dim)
    v = packed_qkv[:, :, q_dim + kv_dim :].view(batch, seq_len, kv_heads, head_dim)
    return apply_interleaved_pair_rope(q, rope), apply_interleaved_pair_rope(k, rope), v


def make_joint3_case(video_len: int, action_len: int, und_len: int, heads: int, head_dim: int):
    packed_v, v_q_w, v_k_w, freqs_re, freqs_im, _, _ = make_case(1, video_len, heads, head_dim)
    packed_a, a_q_w, a_k_w, _, _, _, _ = make_case(1, action_len, heads, head_dim)
    packed_u, u_q_w, u_k_w, _, _, _, _ = make_case(1, und_len, heads, head_dim)
    dim = heads * head_dim
    qkv_v_bias = (0.02 * torch.randn((3 * dim,), device="cuda", dtype=torch.bfloat16)).contiguous()
    total = video_len + action_len + und_len
    q_cat = torch.empty((1, total, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k_cat = torch.empty_like(q_cat)
    v_cat = torch.empty_like(q_cat)
    return (
        packed_v,
        qkv_v_bias,
        v_q_w,
        v_k_w,
        freqs_re,
        freqs_im,
        packed_a,
        a_q_w,
        a_k_w,
        packed_u,
        u_q_w,
        u_k_w,
        q_cat,
        k_cat,
        v_cat,
    )


def torch_ref_joint3(
    packed_v,
    qkv_v_bias,
    v_q_w,
    v_k_w,
    freqs_re,
    freqs_im,
    packed_a,
    a_q_w,
    a_k_w,
    packed_u,
    u_q_w,
    u_k_w,
    heads,
    head_dim,
    eps,
):
    qv, kv, vv = torch_ref_bias(packed_v, qkv_v_bias, v_q_w, v_k_w, freqs_re, freqs_im, heads, head_dim, eps)
    qa, ka, va = torch_ref_no_rope(packed_a, a_q_w, a_k_w, heads, head_dim, eps)
    qu, ku, vu = torch_ref_no_rope(packed_u, u_q_w, u_k_w, heads, head_dim, eps)
    return torch.cat([qv, qa, qu], dim=1), torch.cat([kv, ka, ku], dim=1), torch.cat([vv, va, vu], dim=1)


def time_us(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def percentile(x: torch.Tensor, q: float) -> torch.Tensor:
    flat = x.flatten()
    k = max(1, min(flat.numel(), math.ceil(q * flat.numel())))
    return flat.kthvalue(k).values


def metrics(got, expected):
    diff = (got.float() - expected.float()).abs().flatten()
    return float(percentile(diff, 0.99).item()), float(
        torch.nn.functional.cosine_similarity(got.float().flatten(), expected.float().flatten(), dim=0).item()
    )


def run_one(ops, name: str, shape: tuple[int, int, int, int], args) -> Result:
    batch, seq_len, heads, head_dim = shape
    packed, q_w, k_w, freqs_re, freqs_im, q_out, k_out = make_case(*shape)
    eps = args.eps
    got_q, got_k = ops.qkv_split_norm_rope_bf16(
        packed, q_w, k_w, freqs_re, freqs_im, heads, head_dim, seq_len, eps, q_out, k_out
    )
    exp_q, exp_k = torch_ref(packed, q_w, k_w, freqs_re, freqs_im, heads, head_dim, eps)
    q_p99, q_cos = metrics(got_q, exp_q)
    k_p99, k_cos = metrics(got_k, exp_k)
    flashrt_us = time_us(
        lambda: ops.qkv_split_norm_rope_bf16(
            packed, q_w, k_w, freqs_re, freqs_im, heads, head_dim, seq_len, eps, q_out, k_out
        ),
        args.warmup,
        args.iters,
    )
    eager_us = time_us(
        lambda: torch_ref(packed, q_w, k_w, freqs_re, freqs_im, heads, head_dim, eps),
        args.warmup,
        args.iters,
    )
    status = "PASS" if q_p99 <= args.p99_abs_limit and k_p99 <= args.p99_abs_limit else "FAIL"
    return Result(
        shape=name,
        batch=batch,
        seq_len=seq_len,
        heads=heads,
        head_dim=head_dim,
        flashrt_us=flashrt_us,
        torch_eager_us=eager_us,
        speedup_vs_eager=eager_us / flashrt_us,
        q_p99_abs=q_p99,
        k_p99_abs=k_p99,
        q_cosine=q_cos,
        k_cosine=k_cos,
        status=status,
    )


def run_joint3(ops, name: str, video_len: int, action_len: int, und_len: int, heads: int, head_dim: int, args) -> Result:
    case = make_joint3_case(video_len, action_len, und_len, heads, head_dim)
    (
        packed_v,
        qkv_v_bias,
        v_q_w,
        v_k_w,
        freqs_re,
        freqs_im,
        packed_a,
        a_q_w,
        a_k_w,
        packed_u,
        u_q_w,
        u_k_w,
        q_cat,
        k_cat,
        v_cat,
    ) = case
    eps = args.eps
    got_q, got_k, _ = ops.qkv_split_joint3_cat_bf16(
        packed_v,
        qkv_v_bias,
        v_q_w,
        v_k_w,
        freqs_re,
        freqs_im,
        packed_a,
        a_q_w,
        a_k_w,
        packed_u,
        u_q_w,
        u_k_w,
        heads,
        head_dim,
        q_cat,
        k_cat,
        v_cat,
        video_len,
        eps,
        eps,
        eps,
    )
    exp_q, exp_k, _ = torch_ref_joint3(
        packed_v,
        qkv_v_bias,
        v_q_w,
        v_k_w,
        freqs_re,
        freqs_im,
        packed_a,
        a_q_w,
        a_k_w,
        packed_u,
        u_q_w,
        u_k_w,
        heads,
        head_dim,
        eps,
    )
    q_p99, q_cos = metrics(got_q, exp_q)
    k_p99, k_cos = metrics(got_k, exp_k)
    flashrt_us = time_us(
        lambda: ops.qkv_split_joint3_cat_bf16(
            packed_v,
            qkv_v_bias,
            v_q_w,
            v_k_w,
            freqs_re,
            freqs_im,
            packed_a,
            a_q_w,
            a_k_w,
            packed_u,
            u_q_w,
            u_k_w,
            heads,
            head_dim,
            q_cat,
            k_cat,
            v_cat,
            video_len,
            eps,
            eps,
            eps,
        ),
        args.warmup,
        args.iters,
    )
    eager_us = time_us(
        lambda: torch_ref_joint3(
            packed_v,
            qkv_v_bias,
            v_q_w,
            v_k_w,
            freqs_re,
            freqs_im,
            packed_a,
            a_q_w,
            a_k_w,
            packed_u,
            u_q_w,
            u_k_w,
            heads,
            head_dim,
            eps,
        ),
        args.warmup,
        args.iters,
    )
    status = "PASS" if q_p99 <= args.p99_abs_limit and k_p99 <= args.p99_abs_limit else "FAIL"
    return Result(
        shape=name,
        batch=1,
        seq_len=video_len + action_len + und_len,
        heads=heads,
        head_dim=head_dim,
        flashrt_us=flashrt_us,
        torch_eager_us=eager_us,
        speedup_vs_eager=eager_us / flashrt_us,
        q_p99_abs=q_p99,
        k_p99_abs=k_p99,
        q_cosine=q_cos,
        k_cosine=k_cos,
        status=status,
    )


def run_decode_q(ops, name: str, heads: int, args) -> Result:
    q, _, _, q_w, _, cos, sin = make_decode_case(heads)
    q_out = torch.empty_like(q)
    eps = args.eps
    got = ops.decode_q_norm_rope_stage_bf16(q, q_w, cos, sin, eps, q_out)
    exp = torch_ref_decode(q, q_w, cos, sin, eps)
    q_p99, q_cos = metrics(got, exp)
    flashrt_us = time_us(
        lambda: ops.decode_q_norm_rope_stage_bf16(q, q_w, cos, sin, eps, q_out),
        args.warmup,
        args.iters,
    )
    eager_us = time_us(lambda: torch_ref_decode(q, q_w, cos, sin, eps), args.warmup, args.iters)
    status = "PASS" if q_p99 <= args.p99_abs_limit else "FAIL"
    return Result(
        shape=name,
        batch=1,
        seq_len=1,
        heads=heads,
        head_dim=128,
        flashrt_us=flashrt_us,
        torch_eager_us=eager_us,
        speedup_vs_eager=eager_us / flashrt_us,
        q_p99_abs=q_p99,
        k_p99_abs=0.0,
        q_cosine=q_cos,
        k_cosine=1.0,
        status=status,
    )


def run_decode_kv(ops, name: str, heads: int, devpos: bool, args) -> Result:
    _, k, v, _, k_w, cos, sin = make_decode_case(heads)
    k_slot = torch.empty_like(k)
    v_slot = torch.empty_like(v)
    eps = args.eps
    exp_k = torch_ref_decode(k, k_w, cos, sin, eps)
    if devpos:
        pos = 3
        k_cache = torch.empty((8, heads, 128), device="cuda", dtype=torch.bfloat16)
        v_cache = torch.empty_like(k_cache)
        cur_pos = torch.tensor([pos], device="cuda", dtype=torch.int32)

        def flashrt_fn():
            return ops.decode_k_norm_rope_kvwrite_devpos_bf16(k, v, k_w, cos, sin, cur_pos, k_cache, v_cache, eps)

        def eager_fn():
            k_cache[pos].copy_(torch_ref_decode(k, k_w, cos, sin, eps))
            v_cache[pos].copy_(v)
            return k_cache, v_cache

        flashrt_fn()
        got_k = k_cache[pos]
        got_v = v_cache[pos]
    else:
        def flashrt_fn():
            return ops.decode_k_norm_rope_kvwrite_bf16(k, v, k_w, cos, sin, eps, k_slot, v_slot)

        def eager_fn():
            k_slot.copy_(torch_ref_decode(k, k_w, cos, sin, eps))
            v_slot.copy_(v)
            return k_slot, v_slot

        got_k, got_v = flashrt_fn()
    k_p99, k_cos = metrics(got_k, exp_k)
    v_p99, v_cos = metrics(got_v, v)
    flashrt_us = time_us(flashrt_fn, args.warmup, args.iters)
    eager_us = time_us(eager_fn, args.warmup, args.iters)
    status = "PASS" if k_p99 <= args.p99_abs_limit and v_p99 == 0.0 else "FAIL"
    return Result(
        shape=name,
        batch=1,
        seq_len=1,
        heads=heads,
        head_dim=128,
        flashrt_us=flashrt_us,
        torch_eager_us=eager_us,
        speedup_vs_eager=eager_us / flashrt_us,
        q_p99_abs=v_p99,
        k_p99_abs=k_p99,
        q_cosine=v_cos,
        k_cosine=k_cos,
        status=status,
    )


def run_kvcache_gqa(
    ops,
    name: str,
    batch: int,
    seq_len: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    args,
) -> Result:
    qkv_dim = (q_heads + 2 * kv_heads) * head_dim
    packed = torch.randn((batch, seq_len, qkv_dim), device="cuda", dtype=torch.bfloat16)
    rope = make_interleaved_rope(seq_len, head_dim)
    cache_offset = 2
    max_seq_len = cache_offset + seq_len + 2
    q_out = torch.empty((batch, seq_len, q_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k_cache = torch.empty((batch, max_seq_len, kv_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    got_q, got_k_cache, got_v_cache = ops.qkv_split_rope_kvcache_bf16(
        packed,
        rope,
        q_heads,
        kv_heads,
        head_dim,
        cache_offset,
        q_out,
        k_cache,
        v_cache,
    )
    exp_q, exp_k, exp_v = torch_ref_kvcache(packed, rope, q_heads, kv_heads, head_dim)
    sl = slice(cache_offset, cache_offset + seq_len)
    q_p99, q_cos = metrics(got_q, exp_q)
    k_p99, k_cos = metrics(got_k_cache[:, sl], exp_k)
    v_p99, v_cos = metrics(got_v_cache[:, sl], exp_v)

    def flashrt_fn():
        return ops.qkv_split_rope_kvcache_bf16(
            packed,
            rope,
            q_heads,
            kv_heads,
            head_dim,
            cache_offset,
            q_out,
            k_cache,
            v_cache,
        )

    def eager_fn():
        exp_q_local, exp_k_local, exp_v_local = torch_ref_kvcache(packed, rope, q_heads, kv_heads, head_dim)
        q_out.copy_(exp_q_local)
        k_cache[:, sl].copy_(exp_k_local)
        v_cache[:, sl].copy_(exp_v_local)
        return q_out, k_cache, v_cache

    flashrt_us = time_us(flashrt_fn, args.warmup, args.iters)
    eager_us = time_us(eager_fn, args.warmup, args.iters)
    status = (
        "PASS"
        if q_p99 <= args.p99_abs_limit and k_p99 <= args.p99_abs_limit and v_p99 == 0.0
        else "FAIL"
    )
    return Result(
        shape=name,
        batch=batch,
        seq_len=seq_len,
        heads=q_heads,
        head_dim=head_dim,
        flashrt_us=flashrt_us,
        torch_eager_us=eager_us,
        speedup_vs_eager=eager_us / flashrt_us,
        q_p99_abs=max(q_p99, v_p99),
        k_p99_abs=k_p99,
        q_cosine=min(q_cos, v_cos),
        k_cosine=k_cos,
        status=status,
    )


def write_markdown(path: Path, results: list[Result]) -> None:
    lines = [
        "| Shape | B,L,H,D | FlashRT us | Eager us | vs eager | Q p99 | K p99 | Q cosine | K cosine | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.shape} | {r.batch},{r.seq_len},{r.heads},{r.head_dim} | "
            f"{r.flashrt_us:.3f} | {r.torch_eager_us:.3f} | {r.speedup_vs_eager:.2f}x | "
            f"{r.q_p99_abs:.6f} | {r.k_p99_abs:.6f} | {r.q_cosine:.8f} | "
            f"{r.k_cosine:.8f} | {r.status} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["source", "installed"], default="source")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--shapes", choices=sorted(SHAPE_GROUPS), default="smoke")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--p99-abs-limit", type=float, default=0.015625)
    parser.add_argument("--output", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(37)
    ops = load_source_ops() if args.backend == "source" else load_installed_ops(args.artifact)
    if torch.version.hip is not None:
        results = [
            run_kvcache_gqa(
                ops, "pi05_decoder_gqa_kvcache", 1, 10, 8, 1, 256, args
            ),
            run_kvcache_gqa(
                ops, "pi05_prefix_gqa_kvcache", 1, 712, 8, 1, 256, args
            ),
            run_kvcache_gqa(
                ops, "gqa_batch2_kvcache", 2, 16, 8, 2, 128, args
            ),
        ]
    else:
        results = [
            run_one(ops, name, SHAPES[name], args)
            for name in SHAPE_GROUPS[args.shapes]
        ]
    if torch.version.hip is None and args.shapes in ("smoke", "all"):
        results.append(run_joint3(ops, "joint3_small", 64, 8, 4, 8, 128, args))
        results.append(run_kvcache_gqa(ops, "pi05_decoder_gqa_kvcache", 1, 10, 8, 1, 256, args))
        results.append(run_kvcache_gqa(ops, "pi05_thor_decoder_gqa_kvcache", 1, 50, 8, 1, 256, args))
    if torch.version.hip is None and args.shapes in ("headline", "all"):
        results.append(run_joint3(ops, "joint3_vla", 2520, 16, 16, 24, 128, args))
        results.append(run_decode_q(ops, "decode_q_stage_h24", 24, args))
        results.append(run_decode_kv(ops, "decode_kvwrite_h8", 8, False, args))
        results.append(run_decode_kv(ops, "decode_kvwrite_devpos_h8", 8, True, args))
    if torch.version.hip is None and args.shapes == "headline":
        results.append(run_kvcache_gqa(ops, "pi05_decoder_gqa_kvcache", 1, 10, 8, 1, 256, args))
        results.append(run_kvcache_gqa(ops, "pi05_thor_decoder_gqa_kvcache", 1, 50, 8, 1, 256, args))

    for r in results:
        print(
            f"{r.status} {r.shape}: flashrt={r.flashrt_us:.3f}us "
            f"eager={r.torch_eager_us:.3f}us speedup={r.speedup_vs_eager:.2f}x "
            f"q_p99={r.q_p99_abs:.6f} k_p99={r.k_p99_abs:.6f}"
        )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        write_markdown(Path(args.markdown), results)

    if any(r.status != "PASS" for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
