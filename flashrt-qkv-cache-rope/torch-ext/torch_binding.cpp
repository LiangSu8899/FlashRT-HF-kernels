// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <torch/library.h>

#include <limits>

#if defined(CUDA_KERNEL)
// This package always builds the additive Cosmos3-Edge QK/RoPE surface on
// CUDA. kernel-builder compiles the Torch binding separately from native
// targets, so target-local flags are not visible while this translation unit
// parses the public declarations.
#define FLASHRT_HAVE_COSMOS3_EDGE 1
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "qkv_cache_rope.cuh"
#include "cosmos_edge/cosmos3_edge_misc.cuh"
#elif defined(ROCM_KERNEL)
#include <c10/core/DeviceGuard.h>
#include <c10/hip/HIPStream.h>
#include "rocm/qkv_cache_rope_rocm.h"
#endif

#include "registration.h"
#include "torch_binding.h"

namespace {

void check_cuda_contiguous(torch::Tensor const& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_bf16(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must have dtype torch.bfloat16");
}

void check_fp16(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype torch.float16");
}

void check_fp32(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32,
              name, " must have dtype torch.float32");
}

void check_int32(torch::Tensor const& tensor, const char* name) {
  check_cuda_contiguous(tensor, name);
  TORCH_CHECK(tensor.scalar_type() == torch::kInt32,
              name, " must have dtype torch.int32");
}

int checked_int(int64_t value, const char* name) {
  TORCH_CHECK(value > 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit in positive int");
  return static_cast<int>(value);
}

int checked_nonnegative_int(int64_t value, const char* name) {
  TORCH_CHECK(value >= 0 && value <= std::numeric_limits<int>::max(),
              name, " must fit in non-negative int");
  return static_cast<int>(value);
}

void check_same_device(torch::Tensor const& a,
                       torch::Tensor const& b,
                       const char* a_name,
                       const char* b_name) {
  TORCH_CHECK(a.get_device() == b.get_device(),
              a_name, " and ", b_name, " must be on the same CUDA device");
}

void check_qkv_common(
    torch::Tensor const& packed_qkv,
    torch::Tensor const& norm_q_weight,
    torch::Tensor const& norm_k_weight,
    int64_t heads,
    int64_t head_dim,
    const char* packed_name) {
  TORCH_CHECK(packed_qkv.dim() == 3,
              packed_name, " must have shape (batch, seq_len, 3 * heads * head_dim)");
  TORCH_CHECK(packed_qkv.size(0) > 0 && packed_qkv.size(1) > 0,
              packed_name, " batch and seq_len must be positive");
  TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
  const int64_t dim = heads * head_dim;
  TORCH_CHECK(dim > 0 && dim % 2 == 0, "heads * head_dim must be positive and even");
  TORCH_CHECK(packed_qkv.size(2) == 3 * dim,
              packed_name, ".shape[2] must be 3 * heads * head_dim");
  TORCH_CHECK(norm_q_weight.dim() == 1 && norm_q_weight.size(0) == dim,
              "norm_q_weight must have shape (heads * head_dim,)");
  TORCH_CHECK(norm_k_weight.dim() == 1 && norm_k_weight.size(0) == dim,
              "norm_k_weight must have shape (heads * head_dim,)");
}

void check_freqs(
    torch::Tensor const& freqs_re,
    torch::Tensor const& freqs_im,
    int64_t head_dim,
    int64_t rope_seq_len) {
  check_fp32(freqs_re, "freqs_re");
  check_fp32(freqs_im, "freqs_im");
  TORCH_CHECK(freqs_re.dim() == 2 && freqs_re.size(1) == head_dim / 2,
              "freqs_re must have shape (rope_seq_len, head_dim / 2)");
  TORCH_CHECK(freqs_im.dim() == 2 && freqs_im.sizes() == freqs_re.sizes(),
              "freqs_im must have the same shape as freqs_re");
  TORCH_CHECK(rope_seq_len >= 0,
              "rope_seq_len must be non-negative");
  TORCH_CHECK(freqs_re.size(0) >= rope_seq_len,
              "freqs_re must have at least rope_seq_len rows");
}

void check_cat_out(
    torch::Tensor const& out,
    int64_t batch,
    int64_t total_seq_len,
    int64_t heads,
    int64_t head_dim,
    const char* name) {
  check_bf16(out, name);
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({batch, total_seq_len, heads, head_dim}),
              name, " must have shape (batch, total_seq_len, heads, head_dim)");
}

}  // namespace

void decode_q_norm_rope_stage_bf16(
    torch::Tensor const& q_pre,
    torch::Tensor const& q_norm_weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    double eps,
    torch::Tensor& q_out) {
  check_bf16(q_pre, "q_pre");
  check_bf16(q_norm_weight, "q_norm_weight");
  check_bf16(cos, "cos");
  check_bf16(sin, "sin");
  check_bf16(q_out, "q_out");
  TORCH_CHECK(q_pre.dim() == 2 && q_pre.size(1) == 128,
              "q_pre must have shape (n_q_heads, 128)");
  const int64_t n_q_heads = q_pre.size(0);
  TORCH_CHECK(n_q_heads > 0, "n_q_heads must be positive");
  TORCH_CHECK(q_norm_weight.sizes() == torch::IntArrayRef({128}),
              "q_norm_weight must have shape (128,)");
  TORCH_CHECK(cos.sizes() == torch::IntArrayRef({64}) &&
              sin.sizes() == torch::IntArrayRef({64}),
              "cos and sin must have shape (64,)");
  TORCH_CHECK(q_out.sizes() == q_pre.sizes(),
              "q_out must have shape (n_q_heads, 128)");
  check_same_device(q_pre, q_norm_weight, "q_pre", "q_norm_weight");
  check_same_device(q_pre, cos, "q_pre", "cos");
  check_same_device(q_pre, sin, "q_pre", "sin");
  check_same_device(q_pre, q_out, "q_pre", "q_out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(q_pre.device());
  auto stream = at::cuda::getCurrentCUDAStream(q_pre.get_device()).stream();
  flash_rt::qkv_cache_rope::decode_q_norm_rope_stage_bf16(
      q_pre.data_ptr(),
      q_norm_weight.data_ptr(),
      cos.data_ptr(),
      sin.data_ptr(),
      q_out.data_ptr(),
      checked_int(n_q_heads, "n_q_heads"),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void decode_k_norm_rope_kvwrite_bf16(
    torch::Tensor const& k_pre,
    torch::Tensor const& v_pre,
    torch::Tensor const& k_norm_weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    double eps,
    torch::Tensor& k_cache_dst,
    torch::Tensor& v_cache_dst) {
  check_bf16(k_pre, "k_pre");
  check_bf16(v_pre, "v_pre");
  check_bf16(k_norm_weight, "k_norm_weight");
  check_bf16(cos, "cos");
  check_bf16(sin, "sin");
  check_bf16(k_cache_dst, "k_cache_dst");
  check_bf16(v_cache_dst, "v_cache_dst");
  TORCH_CHECK(k_pre.dim() == 2 && k_pre.size(1) == 128,
              "k_pre must have shape (n_kv_heads, 128)");
  const int64_t n_kv_heads = k_pre.size(0);
  TORCH_CHECK(n_kv_heads > 0, "n_kv_heads must be positive");
  TORCH_CHECK(v_pre.sizes() == k_pre.sizes(),
              "v_pre must have shape (n_kv_heads, 128)");
  TORCH_CHECK(k_norm_weight.sizes() == torch::IntArrayRef({128}),
              "k_norm_weight must have shape (128,)");
  TORCH_CHECK(cos.sizes() == torch::IntArrayRef({64}) &&
              sin.sizes() == torch::IntArrayRef({64}),
              "cos and sin must have shape (64,)");
  TORCH_CHECK(k_cache_dst.sizes() == k_pre.sizes() &&
              v_cache_dst.sizes() == k_pre.sizes(),
              "k_cache_dst and v_cache_dst must have shape (n_kv_heads, 128)");
  check_same_device(k_pre, v_pre, "k_pre", "v_pre");
  check_same_device(k_pre, k_norm_weight, "k_pre", "k_norm_weight");
  check_same_device(k_pre, cos, "k_pre", "cos");
  check_same_device(k_pre, sin, "k_pre", "sin");
  check_same_device(k_pre, k_cache_dst, "k_pre", "k_cache_dst");
  check_same_device(k_pre, v_cache_dst, "k_pre", "v_cache_dst");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(k_pre.device());
  auto stream = at::cuda::getCurrentCUDAStream(k_pre.get_device()).stream();
  flash_rt::qkv_cache_rope::decode_k_norm_rope_kvwrite_bf16(
      k_pre.data_ptr(),
      v_pre.data_ptr(),
      k_norm_weight.data_ptr(),
      cos.data_ptr(),
      sin.data_ptr(),
      k_cache_dst.data_ptr(),
      v_cache_dst.data_ptr(),
      checked_int(n_kv_heads, "n_kv_heads"),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void decode_k_norm_rope_kvwrite_devpos_bf16(
    torch::Tensor const& k_pre,
    torch::Tensor const& v_pre,
    torch::Tensor const& k_norm_weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    torch::Tensor const& cur_pos,
    double eps,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache) {
  check_bf16(k_pre, "k_pre");
  check_bf16(v_pre, "v_pre");
  check_bf16(k_norm_weight, "k_norm_weight");
  check_bf16(cos, "cos");
  check_bf16(sin, "sin");
  check_int32(cur_pos, "cur_pos");
  check_bf16(k_cache, "k_cache");
  check_bf16(v_cache, "v_cache");
  TORCH_CHECK(k_pre.dim() == 2 && k_pre.size(1) == 128,
              "k_pre must have shape (n_kv_heads, 128)");
  const int64_t n_kv_heads = k_pre.size(0);
  TORCH_CHECK(n_kv_heads > 0, "n_kv_heads must be positive");
  TORCH_CHECK(v_pre.sizes() == k_pre.sizes(),
              "v_pre must have shape (n_kv_heads, 128)");
  TORCH_CHECK(k_norm_weight.sizes() == torch::IntArrayRef({128}),
              "k_norm_weight must have shape (128,)");
  TORCH_CHECK(cos.sizes() == torch::IntArrayRef({64}) &&
              sin.sizes() == torch::IntArrayRef({64}),
              "cos and sin must have shape (64,)");
  TORCH_CHECK(cur_pos.numel() == 1, "cur_pos must have one int32 element");
  TORCH_CHECK(k_cache.dim() == 3 &&
              k_cache.size(1) == n_kv_heads &&
              k_cache.size(2) == 128,
              "k_cache must have shape (max_seq_len, n_kv_heads, 128)");
  TORCH_CHECK(v_cache.sizes() == k_cache.sizes(),
              "v_cache must have the same shape as k_cache");
  check_same_device(k_pre, v_pre, "k_pre", "v_pre");
  check_same_device(k_pre, k_norm_weight, "k_pre", "k_norm_weight");
  check_same_device(k_pre, cos, "k_pre", "cos");
  check_same_device(k_pre, sin, "k_pre", "sin");
  check_same_device(k_pre, cur_pos, "k_pre", "cur_pos");
  check_same_device(k_pre, k_cache, "k_pre", "k_cache");
  check_same_device(k_pre, v_cache, "k_pre", "v_cache");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(k_pre.device());
  auto stream = at::cuda::getCurrentCUDAStream(k_pre.get_device()).stream();
  flash_rt::qkv_cache_rope::decode_k_norm_rope_kvwrite_devpos_bf16(
      k_pre.data_ptr(),
      v_pre.data_ptr(),
      k_norm_weight.data_ptr(),
      cos.data_ptr(),
      sin.data_ptr(),
      k_cache.data_ptr(),
      v_cache.data_ptr(),
      cur_pos.data_ptr(),
      checked_int(n_kv_heads * 128, "row_elems"),
      checked_int(n_kv_heads, "n_kv_heads"),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void qkv_split_per_head_norm_rope_bf16(
    torch::Tensor const& packed_qkv,
    torch::Tensor const& q_norm_weight,
    torch::Tensor const& k_norm_weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    int64_t q_heads64,
    int64_t kv_heads64,
    double eps,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    torch::Tensor& v_out) {
  check_bf16(packed_qkv, "packed_qkv");
  check_bf16(q_norm_weight, "q_norm_weight");
  check_bf16(k_norm_weight, "k_norm_weight");
  check_bf16(cos, "cos");
  check_bf16(sin, "sin");
  check_bf16(q_out, "q_out");
  check_bf16(k_out, "k_out");
  check_bf16(v_out, "v_out");
  TORCH_CHECK(packed_qkv.dim() == 3,
              "packed_qkv must have shape "
              "(batch, seq_len, (q_heads + 2 * kv_heads) * 128)");
  const int64_t batch = packed_qkv.size(0);
  const int64_t seq_len = packed_qkv.size(1);
  const int q_heads = checked_int(q_heads64, "q_heads");
  const int kv_heads = checked_int(kv_heads64, "kv_heads");
  constexpr int64_t head_dim = 128;
  const int64_t q_dim = q_heads64 * head_dim;
  const int64_t kv_dim = kv_heads64 * head_dim;
  TORCH_CHECK(batch > 0 && seq_len > 0,
              "batch and seq_len must be positive");
  TORCH_CHECK(packed_qkv.size(2) == q_dim + 2 * kv_dim,
              "packed_qkv.shape[2] must be "
              "(q_heads + 2 * kv_heads) * 128");
  TORCH_CHECK(q_norm_weight.sizes() == torch::IntArrayRef({head_dim}) &&
              k_norm_weight.sizes() == torch::IntArrayRef({head_dim}),
              "Q/K norm weights must have shape (128,)");
  TORCH_CHECK(cos.sizes() ==
                  torch::IntArrayRef({batch, seq_len, head_dim}) &&
              sin.sizes() == cos.sizes(),
              "cos and sin must have shape (batch, seq_len, 128)");
  TORCH_CHECK(q_out.sizes() ==
                  torch::IntArrayRef({batch, seq_len, q_heads64, head_dim}),
              "q_out must have shape (batch, seq_len, q_heads, 128)");
  TORCH_CHECK(k_out.sizes() ==
                  torch::IntArrayRef({batch, seq_len, kv_heads64, head_dim}) &&
              v_out.sizes() == k_out.sizes(),
              "k_out and v_out must have shape "
              "(batch, seq_len, kv_heads, 128)");
  check_same_device(
      packed_qkv, q_norm_weight, "packed_qkv", "q_norm_weight");
  check_same_device(
      packed_qkv, k_norm_weight, "packed_qkv", "k_norm_weight");
  check_same_device(packed_qkv, cos, "packed_qkv", "cos");
  check_same_device(packed_qkv, sin, "packed_qkv", "sin");
  check_same_device(packed_qkv, q_out, "packed_qkv", "q_out");
  check_same_device(packed_qkv, k_out, "packed_qkv", "k_out");
  check_same_device(packed_qkv, v_out, "packed_qkv", "v_out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed_qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream(
      packed_qkv.get_device()).stream();
  flash_rt::qkv_cache_rope::qkv_split_per_head_norm_rope_bf16(
      packed_qkv.data_ptr(),
      q_norm_weight.data_ptr(),
      k_norm_weight.data_ptr(),
      cos.data_ptr(),
      sin.data_ptr(),
      q_out.data_ptr(),
      k_out.data_ptr(),
      v_out.data_ptr(),
      checked_int(batch * seq_len, "rows"),
      q_heads,
      kv_heads,
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void qkv_split_bias_rope_bf16(
    torch::Tensor const& packed_qkv,
    torch::Tensor const& qkv_bias,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    int64_t q_heads64,
    int64_t kv_heads64,
    int64_t head_dim64,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    torch::Tensor& v_out) {
  check_bf16(packed_qkv, "packed_qkv");
  check_bf16(qkv_bias, "qkv_bias");
  check_fp32(cos, "cos");
  check_fp32(sin, "sin");
  check_bf16(q_out, "q_out");
  check_bf16(k_out, "k_out");
  check_bf16(v_out, "v_out");
  TORCH_CHECK(packed_qkv.dim() == 3,
              "packed_qkv must have shape "
              "(batch, seq_len, (q_heads + 2 * kv_heads) * head_dim)");
  const int64_t batch = packed_qkv.size(0);
  const int64_t seq_len = packed_qkv.size(1);
  const int64_t q_dim = q_heads64 * head_dim64;
  const int64_t kv_dim = kv_heads64 * head_dim64;
  const int64_t packed_dim = q_dim + 2 * kv_dim;
  TORCH_CHECK(batch > 0 && seq_len > 0,
              "batch and seq_len must be positive");
  TORCH_CHECK(q_heads64 > 0 && kv_heads64 > 0,
              "q_heads and kv_heads must be positive");
  TORCH_CHECK(head_dim64 >= 8 && head_dim64 <= 256 &&
                  head_dim64 % 2 == 0,
              "head_dim must be even and in [8, 256]");
  TORCH_CHECK(packed_qkv.size(2) == packed_dim,
              "packed_qkv has an invalid QKV width");
  TORCH_CHECK(qkv_bias.sizes() == torch::IntArrayRef({packed_dim}),
              "qkv_bias must have shape "
              "((q_heads + 2 * kv_heads) * head_dim,)");
  TORCH_CHECK(cos.dim() == 3 &&
                  cos.size(0) == batch && cos.size(1) == seq_len &&
                  (cos.size(2) == head_dim64 / 2 ||
                   cos.size(2) == head_dim64),
              "cos must have shape (batch, seq_len, head_dim / 2) "
              "or (batch, seq_len, head_dim)");
  TORCH_CHECK(sin.sizes() == cos.sizes(),
              "sin must have the same shape as cos");
  TORCH_CHECK(q_out.sizes() == torch::IntArrayRef(
                  {batch, seq_len, q_heads64, head_dim64}),
              "q_out has an invalid shape");
  TORCH_CHECK(k_out.sizes() == torch::IntArrayRef(
                  {batch, seq_len, kv_heads64, head_dim64}) &&
                  v_out.sizes() == k_out.sizes(),
              "k_out and v_out have an invalid shape");
  check_same_device(packed_qkv, qkv_bias, "packed_qkv", "qkv_bias");
  check_same_device(packed_qkv, cos, "packed_qkv", "cos");
  check_same_device(packed_qkv, sin, "packed_qkv", "sin");
  check_same_device(packed_qkv, q_out, "packed_qkv", "q_out");
  check_same_device(packed_qkv, k_out, "packed_qkv", "k_out");
  check_same_device(packed_qkv, v_out, "packed_qkv", "v_out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed_qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream(
      packed_qkv.get_device()).stream();
  flash_rt::qkv_cache_rope::qkv_split_bias_rope_bf16(
      packed_qkv.data_ptr(),
      qkv_bias.data_ptr(),
      cos.data_ptr<float>(),
      sin.data_ptr<float>(),
      q_out.data_ptr(),
      k_out.data_ptr(),
      v_out.data_ptr(),
      checked_int(batch * seq_len, "rows"),
      checked_int(q_heads64, "q_heads"),
      checked_int(kv_heads64, "kv_heads"),
      checked_int(head_dim64, "head_dim"),
      checked_int(cos.size(2), "rope_stride"),
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void qkv_split_bias_rope_fp16(
    torch::Tensor const& packed_qkv,
    torch::Tensor const& qkv_bias,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    int64_t q_heads64,
    int64_t kv_heads64,
    int64_t head_dim64,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    torch::Tensor& v_out) {
  check_bf16(packed_qkv, "packed_qkv");
  check_bf16(qkv_bias, "qkv_bias");
  check_fp32(cos, "cos");
  check_fp32(sin, "sin");
  check_fp16(q_out, "q_out");
  check_fp16(k_out, "k_out");
  check_fp16(v_out, "v_out");
  TORCH_CHECK(packed_qkv.dim() == 3,
              "packed_qkv must have shape "
              "(batch, seq_len, (q_heads + 2 * kv_heads) * head_dim)");
  const int64_t batch = packed_qkv.size(0);
  const int64_t seq_len = packed_qkv.size(1);
  const int64_t q_dim = q_heads64 * head_dim64;
  const int64_t kv_dim = kv_heads64 * head_dim64;
  const int64_t packed_dim = q_dim + 2 * kv_dim;
  TORCH_CHECK(batch > 0 && seq_len > 0,
              "batch and seq_len must be positive");
  TORCH_CHECK(q_heads64 > 0 && kv_heads64 > 0,
              "q_heads and kv_heads must be positive");
  TORCH_CHECK(head_dim64 >= 8 && head_dim64 <= 256 &&
                  head_dim64 % 2 == 0,
              "head_dim must be even and in [8, 256]");
  TORCH_CHECK(packed_qkv.size(2) == packed_dim,
              "packed_qkv has an invalid QKV width");
  TORCH_CHECK(qkv_bias.sizes() == torch::IntArrayRef({packed_dim}),
              "qkv_bias has an invalid shape");
  TORCH_CHECK(cos.dim() == 3 && cos.size(0) == batch &&
                  cos.size(1) == seq_len &&
                  (cos.size(2) == head_dim64 / 2 ||
                   cos.size(2) == head_dim64),
              "cos has an invalid shape");
  TORCH_CHECK(sin.sizes() == cos.sizes(),
              "sin must have the same shape as cos");
  TORCH_CHECK(q_out.sizes() == torch::IntArrayRef(
                  {batch, seq_len, q_heads64, head_dim64}),
              "q_out has an invalid shape");
  TORCH_CHECK(k_out.sizes() == torch::IntArrayRef(
                  {batch, seq_len, kv_heads64, head_dim64}) &&
                  v_out.sizes() == k_out.sizes(),
              "k_out and v_out have an invalid shape");
  check_same_device(packed_qkv, qkv_bias, "packed_qkv", "qkv_bias");
  check_same_device(packed_qkv, cos, "packed_qkv", "cos");
  check_same_device(packed_qkv, sin, "packed_qkv", "sin");
  check_same_device(packed_qkv, q_out, "packed_qkv", "q_out");
  check_same_device(packed_qkv, k_out, "packed_qkv", "k_out");
  check_same_device(packed_qkv, v_out, "packed_qkv", "v_out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed_qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream(
      packed_qkv.get_device()).stream();
  flash_rt::qkv_cache_rope::qkv_split_bias_rope_fp16(
      packed_qkv.data_ptr(), qkv_bias.data_ptr(), cos.data_ptr<float>(),
      sin.data_ptr<float>(), q_out.data_ptr(), k_out.data_ptr(),
      v_out.data_ptr(), checked_int(batch * seq_len, "rows"),
      checked_int(q_heads64, "q_heads"), checked_int(kv_heads64, "kv_heads"),
      checked_int(head_dim64, "head_dim"),
      checked_int(cos.size(2), "rope_stride"), stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void qkv_split_rope_kvcache_bf16(
    torch::Tensor const& packed_qkv,
    torch::Tensor const& rope,
    int64_t q_heads64,
    int64_t kv_heads64,
    int64_t head_dim64,
    int64_t cache_offset64,
    torch::Tensor& q_out,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache) {
  check_bf16(packed_qkv, "packed_qkv");
  check_bf16(rope, "rope");
  check_bf16(q_out, "q_out");
  check_bf16(k_cache, "k_cache");
  check_bf16(v_cache, "v_cache");
  TORCH_CHECK(packed_qkv.dim() == 3,
              "packed_qkv must have shape (batch, seq_len, (q_heads + 2 * kv_heads) * head_dim)");
  const int64_t batch = packed_qkv.size(0);
  const int64_t seq_len = packed_qkv.size(1);
  const int q_heads = checked_int(q_heads64, "q_heads");
  const int kv_heads = checked_int(kv_heads64, "kv_heads");
  const int head_dim = checked_int(head_dim64, "head_dim");
  TORCH_CHECK(batch > 0 && seq_len > 0, "batch and seq_len must be positive");
  TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
  const int64_t q_dim = static_cast<int64_t>(q_heads) * head_dim;
  const int64_t kv_dim = static_cast<int64_t>(kv_heads) * head_dim;
  TORCH_CHECK(packed_qkv.size(2) == q_dim + 2 * kv_dim,
              "packed_qkv.shape[2] must be (q_heads + 2 * kv_heads) * head_dim");
  TORCH_CHECK(rope.dim() == 2 && rope.size(0) >= seq_len && rope.size(1) == head_dim,
              "rope must have shape (>= seq_len, head_dim)");
  TORCH_CHECK(q_out.sizes() == torch::IntArrayRef({batch, seq_len, q_heads, head_dim}),
              "q_out must have shape (batch, seq_len, q_heads, head_dim)");
  TORCH_CHECK(k_cache.dim() == 4 &&
              k_cache.size(0) == batch &&
              k_cache.size(2) == kv_heads &&
              k_cache.size(3) == head_dim,
              "k_cache must have shape (batch, max_seq_len, kv_heads, head_dim)");
  TORCH_CHECK(v_cache.sizes() == k_cache.sizes(),
              "v_cache must have the same shape as k_cache");
  const int64_t max_seq_len = k_cache.size(1);
  TORCH_CHECK(cache_offset64 >= 0 && cache_offset64 + seq_len <= max_seq_len,
              "cache_offset + seq_len must be within k_cache.shape[1]");
  check_same_device(packed_qkv, rope, "packed_qkv", "rope");
  check_same_device(packed_qkv, q_out, "packed_qkv", "q_out");
  check_same_device(packed_qkv, k_cache, "packed_qkv", "k_cache");
  check_same_device(packed_qkv, v_cache, "packed_qkv", "v_cache");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed_qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed_qkv.get_device()).stream();
  flash_rt::qkv_cache_rope::qkv_split_rope_kvcache_bf16(
      packed_qkv.data_ptr(),
      rope.data_ptr(),
      q_out.data_ptr(),
      k_cache.data_ptr(),
      v_cache.data_ptr(),
      checked_int(batch, "batch"),
      checked_int(seq_len, "seq_len"),
      checked_int(max_seq_len, "max_seq_len"),
      q_heads,
      kv_heads,
      head_dim,
      checked_nonnegative_int(cache_offset64, "cache_offset"),
      stream);
#elif defined(ROCM_KERNEL)
  c10::OptionalDeviceGuard device_guard(packed_qkv.device());
  auto stream = c10::hip::getCurrentHIPStream(
      packed_qkv.get_device()).stream();
  flash_rt::qkv_cache_rope::qkv_split_rope_kvcache_bf16_rocm(
      packed_qkv.data_ptr(),
      rope.data_ptr(),
      q_out.data_ptr(),
      k_cache.data_ptr(),
      v_cache.data_ptr(),
      checked_int(batch, "batch"),
      checked_int(seq_len, "seq_len"),
      checked_int(max_seq_len, "max_seq_len"),
      q_heads,
      kv_heads,
      head_dim,
      checked_nonnegative_int(cache_offset64, "cache_offset"),
      stream);
#else
  TORCH_CHECK(false,
              "qkv_split_rope_kvcache_bf16 requires CUDA or ROCm support");
#endif
}

void qkv_split_rope_kvcache_fp16(
    torch::Tensor const& packed_qkv,
    torch::Tensor const& rope,
    int64_t q_heads64,
    int64_t kv_heads64,
    int64_t head_dim64,
    int64_t cache_offset64,
    c10::optional<torch::Tensor> const& device_position,
    torch::Tensor& q_out,
    torch::Tensor& k_cache,
    torch::Tensor& v_cache) {
  check_fp16(packed_qkv, "packed_qkv");
  check_fp16(rope, "rope");
  check_fp16(q_out, "q_out");
  check_fp16(k_cache, "k_cache");
  check_fp16(v_cache, "v_cache");
  TORCH_CHECK(packed_qkv.dim() == 3,
              "packed_qkv must have shape (batch, seq_len, packed_width)");
  const int64_t batch = packed_qkv.size(0);
  const int64_t seq_len = packed_qkv.size(1);
  const int q_heads = checked_int(q_heads64, "q_heads");
  const int kv_heads = checked_int(kv_heads64, "kv_heads");
  const int head_dim = checked_int(head_dim64, "head_dim");
  TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
  const int64_t q_dim = q_heads64 * head_dim64;
  const int64_t kv_dim = kv_heads64 * head_dim64;
  TORCH_CHECK(packed_qkv.size(2) == q_dim + 2 * kv_dim,
              "packed_qkv has an invalid GQA width");
  TORCH_CHECK(rope.dim() == 2 && rope.size(0) >= seq_len &&
                  rope.size(1) == head_dim,
              "rope must have shape (>= seq_len, head_dim)");
  TORCH_CHECK(q_out.sizes() ==
                  torch::IntArrayRef({batch, seq_len, q_heads64, head_dim64}),
              "q_out has an invalid shape");
  TORCH_CHECK(k_cache.dim() == 4 && k_cache.size(0) == batch &&
                  k_cache.size(2) == kv_heads64 &&
                  k_cache.size(3) == head_dim64,
              "k_cache must have shape (batch, max_seq_len, kv_heads, head_dim)");
  TORCH_CHECK(v_cache.sizes() == k_cache.sizes(),
              "v_cache must have the same shape as k_cache");
  TORCH_CHECK(cache_offset64 >= 0, "cache_offset must be non-negative");
  if (!device_position.has_value()) {
    TORCH_CHECK(cache_offset64 + seq_len <= k_cache.size(1),
                "cache_offset + seq_len exceeds cache capacity");
  }
  check_same_device(packed_qkv, rope, "packed_qkv", "rope");
  check_same_device(packed_qkv, q_out, "packed_qkv", "q_out");
  check_same_device(packed_qkv, k_cache, "packed_qkv", "k_cache");
  check_same_device(packed_qkv, v_cache, "packed_qkv", "v_cache");
  const void* device_position_ptr = nullptr;
  if (device_position.has_value()) {
    auto const& position = device_position.value();
    check_int32(position, "device_position");
    TORCH_CHECK(position.numel() == 1,
                "device_position must contain one int32 element");
    check_same_device(packed_qkv, position, "packed_qkv", "device_position");
    device_position_ptr = position.data_ptr();
  }
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed_qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed_qkv.get_device()).stream();
  flash_rt::qkv_cache_rope::qkv_split_rope_kvcache_fp16(
      packed_qkv.data_ptr(), rope.data_ptr(), q_out.data_ptr(),
      k_cache.data_ptr(), v_cache.data_ptr(), device_position_ptr,
      checked_int(batch, "batch"), checked_int(seq_len, "seq_len"),
      checked_int(k_cache.size(1), "max_seq_len"), q_heads, kv_heads,
      head_dim, checked_nonnegative_int(cache_offset64, "cache_offset"), stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void qk_norm_rope_strided_bf16(
    torch::Tensor const& q_in,
    torch::Tensor const& k_in,
    torch::Tensor const& q_weight,
    torch::Tensor const& k_weight,
    torch::Tensor const& cos,
    torch::Tensor const& sin,
    int64_t q_heads64,
    int64_t k_heads64,
    double eps,
    torch::Tensor& q_out,
    torch::Tensor& k_out) {
  check_bf16(q_in, "q_in");
  check_bf16(k_in, "k_in");
  check_bf16(q_weight, "q_weight");
  check_bf16(k_weight, "k_weight");
  check_bf16(cos, "cos");
  check_bf16(sin, "sin");
  check_bf16(q_out, "q_out");
  check_bf16(k_out, "k_out");
  const int q_heads = checked_int(q_heads64, "q_heads");
  const int k_heads = checked_int(k_heads64, "k_heads");
  TORCH_CHECK(q_in.dim() == 2 && k_in.dim() == 2 &&
                  q_in.size(0) == k_in.size(0),
              "q_in and k_in must be rank-2 with the same row count");
  const int64_t rows = q_in.size(0);
  TORCH_CHECK(q_in.size(1) >= q_heads64 * 128 &&
                  k_in.size(1) >= k_heads64 * 128,
              "input row widths must cover heads * 128");
  TORCH_CHECK(q_weight.sizes() == torch::IntArrayRef({128}) &&
                  k_weight.sizes() == torch::IntArrayRef({128}),
              "Q/K weights must have shape (128,)");
  TORCH_CHECK(cos.sizes() == torch::IntArrayRef({rows, 128}) &&
                  sin.sizes() == cos.sizes(),
              "cos and sin must have shape (rows, 128)");
  TORCH_CHECK(q_out.sizes() == torch::IntArrayRef({rows, q_heads64, 128}) &&
                  k_out.sizes() == torch::IntArrayRef({rows, k_heads64, 128}),
              "output shapes must be (rows, heads, 128)");
  check_same_device(q_in, k_in, "q_in", "k_in");
  check_same_device(q_in, q_weight, "q_in", "q_weight");
  check_same_device(q_in, k_weight, "q_in", "k_weight");
  check_same_device(q_in, cos, "q_in", "cos");
  check_same_device(q_in, sin, "q_in", "sin");
  check_same_device(q_in, q_out, "q_in", "q_out");
  check_same_device(q_in, k_out, "q_in", "k_out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(q_in.device());
  auto stream = at::cuda::getCurrentCUDAStream(q_in.get_device()).stream();
  flash_rt::kernels::cosmos3_edge_qk_norm_rope_strided_bf16(
      static_cast<const __nv_bfloat16*>(q_in.data_ptr()),
      static_cast<const __nv_bfloat16*>(k_in.data_ptr()),
      static_cast<const __nv_bfloat16*>(q_weight.data_ptr()),
      static_cast<const __nv_bfloat16*>(k_weight.data_ptr()),
      static_cast<const __nv_bfloat16*>(cos.data_ptr()),
      static_cast<const __nv_bfloat16*>(sin.data_ptr()),
      static_cast<__nv_bfloat16*>(q_out.data_ptr()),
      static_cast<__nv_bfloat16*>(k_out.data_ptr()),
      checked_int(rows, "rows"), q_heads, k_heads,
      checked_int(q_in.size(1), "q_input_row_stride"),
      checked_int(k_in.size(1), "k_input_row_stride"),
      static_cast<float>(eps), stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void qkv_split_bf16(
    torch::Tensor const& packed_qkv,
    int64_t heads64,
    int64_t head_dim64,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    torch::Tensor& v_out) {
  check_bf16(packed_qkv, "packed_qkv");
  check_bf16(q_out, "q_out");
  check_bf16(k_out, "k_out");
  check_bf16(v_out, "v_out");
  TORCH_CHECK(packed_qkv.dim() == 3,
              "packed_qkv must have shape (batch, seq_len, 3 * heads * head_dim)");
  const int64_t batch = packed_qkv.size(0);
  const int64_t seq_len = packed_qkv.size(1);
  const int heads = checked_int(heads64, "heads");
  const int head_dim = checked_int(head_dim64, "head_dim");
  TORCH_CHECK(batch > 0 && seq_len > 0, "batch and seq_len must be positive");
  const int64_t dim = static_cast<int64_t>(heads) * head_dim;
  TORCH_CHECK(packed_qkv.size(2) == 3 * dim,
              "packed_qkv.shape[2] must be 3 * heads * head_dim");
  TORCH_CHECK(q_out.sizes() == torch::IntArrayRef({batch, seq_len, heads, head_dim}),
              "q_out must have shape (batch, seq_len, heads, head_dim)");
  TORCH_CHECK(k_out.sizes() == q_out.sizes(),
              "k_out must have the same shape as q_out");
  TORCH_CHECK(v_out.sizes() == q_out.sizes(),
              "v_out must have the same shape as q_out");
  check_same_device(packed_qkv, q_out, "packed_qkv", "q_out");
  check_same_device(packed_qkv, k_out, "packed_qkv", "k_out");
  check_same_device(packed_qkv, v_out, "packed_qkv", "v_out");
#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed_qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed_qkv.get_device()).stream();
  flash_rt::qkv_cache_rope::qkv_split_bf16(
      packed_qkv.data_ptr(),
      q_out.data_ptr(),
      k_out.data_ptr(),
      v_out.data_ptr(),
      checked_int(batch, "batch"),
      checked_int(seq_len, "seq_len"),
      heads,
      head_dim,
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void qkv_split_norm_rope_bf16(
    torch::Tensor const& packed_qkv,
    torch::Tensor const& norm_q_weight,
    torch::Tensor const& norm_k_weight,
    torch::Tensor const& freqs_re,
    torch::Tensor const& freqs_im,
    int64_t heads64,
    int64_t head_dim64,
    int64_t rope_seq_len64,
    double eps,
    torch::Tensor& q_out,
    torch::Tensor& k_out) {
  check_bf16(packed_qkv, "packed_qkv");
  check_bf16(norm_q_weight, "norm_q_weight");
  check_bf16(norm_k_weight, "norm_k_weight");
  check_fp32(freqs_re, "freqs_re");
  check_fp32(freqs_im, "freqs_im");
  check_bf16(q_out, "q_out");
  check_bf16(k_out, "k_out");
  TORCH_CHECK(packed_qkv.dim() == 3,
              "packed_qkv must have shape (batch, seq_len, 3 * heads * head_dim)");
  const int64_t batch = packed_qkv.size(0);
  const int64_t seq_len = packed_qkv.size(1);
  const int heads = checked_int(heads64, "heads");
  const int head_dim = checked_int(head_dim64, "head_dim");
  const int64_t dim = static_cast<int64_t>(heads) * head_dim;
  TORCH_CHECK(batch > 0 && seq_len > 0, "batch and seq_len must be positive");
  TORCH_CHECK(head_dim % 2 == 0, "head_dim must be even");
  TORCH_CHECK(dim % 2 == 0, "heads * head_dim must be even");
  TORCH_CHECK(packed_qkv.size(2) == 3 * dim,
              "packed_qkv.shape[2] must be 3 * heads * head_dim");
  TORCH_CHECK(norm_q_weight.dim() == 1 && norm_q_weight.size(0) == dim,
              "norm_q_weight must have shape (heads * head_dim,)");
  TORCH_CHECK(norm_k_weight.dim() == 1 && norm_k_weight.size(0) == dim,
              "norm_k_weight must have shape (heads * head_dim,)");
  TORCH_CHECK(freqs_re.dim() == 2 && freqs_re.size(1) == head_dim / 2,
              "freqs_re must have shape (rope_seq_len, head_dim / 2)");
  TORCH_CHECK(freqs_im.dim() == 2 && freqs_im.sizes() == freqs_re.sizes(),
              "freqs_im must have the same shape as freqs_re");
  TORCH_CHECK(rope_seq_len64 >= 0 && rope_seq_len64 <= seq_len,
              "rope_seq_len must be in [0, packed_qkv.shape[1]]");
  TORCH_CHECK(freqs_re.size(0) >= rope_seq_len64,
              "freqs_re must have at least rope_seq_len rows");
  TORCH_CHECK(q_out.sizes() == torch::IntArrayRef({batch, seq_len, heads, head_dim}),
              "q_out must have shape (batch, seq_len, heads, head_dim)");
  TORCH_CHECK(k_out.sizes() == q_out.sizes(),
              "k_out must have the same shape as q_out");

  check_same_device(packed_qkv, norm_q_weight, "packed_qkv", "norm_q_weight");
  check_same_device(packed_qkv, norm_k_weight, "packed_qkv", "norm_k_weight");
  check_same_device(packed_qkv, freqs_re, "packed_qkv", "freqs_re");
  check_same_device(packed_qkv, freqs_im, "packed_qkv", "freqs_im");
  check_same_device(packed_qkv, q_out, "packed_qkv", "q_out");
  check_same_device(packed_qkv, k_out, "packed_qkv", "k_out");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed_qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed_qkv.get_device()).stream();
  flash_rt::qkv_cache_rope::qkv_split_norm_rope_bf16(
      packed_qkv.data_ptr(),
      norm_q_weight.data_ptr(),
      norm_k_weight.data_ptr(),
      reinterpret_cast<const float*>(freqs_re.data_ptr()),
      reinterpret_cast<const float*>(freqs_im.data_ptr()),
      q_out.data_ptr(),
      k_out.data_ptr(),
      checked_int(batch, "batch"),
      checked_int(seq_len, "seq_len"),
      heads,
      head_dim,
      checked_int(rope_seq_len64, "rope_seq_len"),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void qkv_split_bias_norm_rope_v_bf16(
    torch::Tensor const& packed_qkv,
    torch::Tensor const& qkv_bias,
    torch::Tensor const& norm_q_weight,
    torch::Tensor const& norm_k_weight,
    torch::Tensor const& freqs_re,
    torch::Tensor const& freqs_im,
    int64_t heads64,
    int64_t head_dim64,
    int64_t rope_seq_len64,
    double eps,
    torch::Tensor& q_out,
    torch::Tensor& k_out,
    torch::Tensor& v_out) {
  check_bf16(packed_qkv, "packed_qkv");
  check_bf16(qkv_bias, "qkv_bias");
  check_bf16(norm_q_weight, "norm_q_weight");
  check_bf16(norm_k_weight, "norm_k_weight");
  check_bf16(q_out, "q_out");
  check_bf16(k_out, "k_out");
  check_bf16(v_out, "v_out");
  const int heads = checked_int(heads64, "heads");
  const int head_dim = checked_int(head_dim64, "head_dim");
  check_qkv_common(packed_qkv, norm_q_weight, norm_k_weight, heads, head_dim, "packed_qkv");
  const int64_t batch = packed_qkv.size(0);
  const int64_t seq_len = packed_qkv.size(1);
  const int64_t dim = static_cast<int64_t>(heads) * head_dim;
  TORCH_CHECK(qkv_bias.dim() == 1 && qkv_bias.size(0) == 3 * dim,
              "qkv_bias must have shape (3 * heads * head_dim,)");
  TORCH_CHECK(rope_seq_len64 >= 0 && rope_seq_len64 <= seq_len,
              "rope_seq_len must be in [0, packed_qkv.shape[1]]");
  check_freqs(freqs_re, freqs_im, head_dim, rope_seq_len64);
  TORCH_CHECK(q_out.sizes() == torch::IntArrayRef({batch, seq_len, heads, head_dim}),
              "q_out must have shape (batch, seq_len, heads, head_dim)");
  TORCH_CHECK(k_out.sizes() == q_out.sizes() && v_out.sizes() == q_out.sizes(),
              "k_out and v_out must have the same shape as q_out");

  check_same_device(packed_qkv, qkv_bias, "packed_qkv", "qkv_bias");
  check_same_device(packed_qkv, norm_q_weight, "packed_qkv", "norm_q_weight");
  check_same_device(packed_qkv, norm_k_weight, "packed_qkv", "norm_k_weight");
  check_same_device(packed_qkv, freqs_re, "packed_qkv", "freqs_re");
  check_same_device(packed_qkv, freqs_im, "packed_qkv", "freqs_im");
  check_same_device(packed_qkv, q_out, "packed_qkv", "q_out");
  check_same_device(packed_qkv, k_out, "packed_qkv", "k_out");
  check_same_device(packed_qkv, v_out, "packed_qkv", "v_out");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed_qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed_qkv.get_device()).stream();
  flash_rt::qkv_cache_rope::qkv_split_bias_norm_rope_v_bf16(
      packed_qkv.data_ptr(),
      qkv_bias.data_ptr(),
      norm_q_weight.data_ptr(),
      norm_k_weight.data_ptr(),
      reinterpret_cast<const float*>(freqs_re.data_ptr()),
      reinterpret_cast<const float*>(freqs_im.data_ptr()),
      q_out.data_ptr(),
      k_out.data_ptr(),
      v_out.data_ptr(),
      checked_int(batch, "batch"),
      checked_int(seq_len, "seq_len"),
      heads,
      head_dim,
      checked_int(rope_seq_len64, "rope_seq_len"),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void qkv_split_bias_norm_rope_v_cat_bf16(
    torch::Tensor const& packed_qkv,
    torch::Tensor const& qkv_bias,
    torch::Tensor const& norm_q_weight,
    torch::Tensor const& norm_k_weight,
    torch::Tensor const& freqs_re,
    torch::Tensor const& freqs_im,
    int64_t heads64,
    int64_t head_dim64,
    int64_t video_offset64,
    int64_t rope_seq_len64,
    double eps,
    torch::Tensor& q_cat_out,
    torch::Tensor& k_cat_out,
    torch::Tensor& v_cat_out) {
  check_bf16(packed_qkv, "packed_qkv");
  check_bf16(qkv_bias, "qkv_bias");
  check_bf16(norm_q_weight, "norm_q_weight");
  check_bf16(norm_k_weight, "norm_k_weight");
  const int heads = checked_int(heads64, "heads");
  const int head_dim = checked_int(head_dim64, "head_dim");
  check_qkv_common(packed_qkv, norm_q_weight, norm_k_weight, heads, head_dim, "packed_qkv");
  const int64_t batch = packed_qkv.size(0);
  const int64_t seq_len = packed_qkv.size(1);
  const int64_t dim = static_cast<int64_t>(heads) * head_dim;
  TORCH_CHECK(qkv_bias.dim() == 1 && qkv_bias.size(0) == 3 * dim,
              "qkv_bias must have shape (3 * heads * head_dim,)");
  TORCH_CHECK(q_cat_out.dim() == 4,
              "q_cat_out must have shape (batch, total_seq_len, heads, head_dim)");
  const int64_t total_seq_len = q_cat_out.size(1);
  TORCH_CHECK(video_offset64 >= 0 && video_offset64 + seq_len <= total_seq_len,
              "video_offset + packed_qkv.shape[1] must be within q_cat_out.shape[1]");
  TORCH_CHECK(rope_seq_len64 >= 0 && rope_seq_len64 <= seq_len,
              "rope_seq_len must be in [0, packed_qkv.shape[1]]");
  check_freqs(freqs_re, freqs_im, head_dim, rope_seq_len64);
  check_cat_out(q_cat_out, batch, total_seq_len, heads, head_dim, "q_cat_out");
  check_cat_out(k_cat_out, batch, total_seq_len, heads, head_dim, "k_cat_out");
  check_cat_out(v_cat_out, batch, total_seq_len, heads, head_dim, "v_cat_out");

  check_same_device(packed_qkv, qkv_bias, "packed_qkv", "qkv_bias");
  check_same_device(packed_qkv, norm_q_weight, "packed_qkv", "norm_q_weight");
  check_same_device(packed_qkv, norm_k_weight, "packed_qkv", "norm_k_weight");
  check_same_device(packed_qkv, freqs_re, "packed_qkv", "freqs_re");
  check_same_device(packed_qkv, freqs_im, "packed_qkv", "freqs_im");
  check_same_device(packed_qkv, q_cat_out, "packed_qkv", "q_cat_out");
  check_same_device(packed_qkv, k_cat_out, "packed_qkv", "k_cat_out");
  check_same_device(packed_qkv, v_cat_out, "packed_qkv", "v_cat_out");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed_qkv.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed_qkv.get_device()).stream();
  flash_rt::qkv_cache_rope::qkv_split_bias_norm_rope_v_cat_bf16(
      packed_qkv.data_ptr(),
      qkv_bias.data_ptr(),
      norm_q_weight.data_ptr(),
      norm_k_weight.data_ptr(),
      reinterpret_cast<const float*>(freqs_re.data_ptr()),
      reinterpret_cast<const float*>(freqs_im.data_ptr()),
      q_cat_out.data_ptr(),
      k_cat_out.data_ptr(),
      v_cat_out.data_ptr(),
      checked_int(batch, "batch"),
      checked_int(total_seq_len, "total_seq_len"),
      checked_int(video_offset64, "video_offset"),
      checked_int(seq_len, "video_seq_len"),
      heads,
      head_dim,
      checked_int(rope_seq_len64, "rope_seq_len"),
      static_cast<float>(eps),
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

void qkv_split_joint3_cat_bf16(
    torch::Tensor const& packed_v,
    torch::Tensor const& qkv_v_bias,
    torch::Tensor const& norm_v_q_weight,
    torch::Tensor const& norm_v_k_weight,
    torch::Tensor const& freqs_re,
    torch::Tensor const& freqs_im,
    torch::Tensor const& packed_a,
    torch::Tensor const& norm_a_q_weight,
    torch::Tensor const& norm_a_k_weight,
    torch::Tensor const& packed_u,
    torch::Tensor const& norm_u_q_weight,
    torch::Tensor const& norm_u_k_weight,
    int64_t heads64,
    int64_t head_dim64,
    int64_t rope_seq_len64,
    double eps_v,
    double eps_a,
    double eps_u,
    torch::Tensor& q_cat_out,
    torch::Tensor& k_cat_out,
    torch::Tensor& v_cat_out) {
  check_bf16(packed_v, "packed_v");
  check_bf16(qkv_v_bias, "qkv_v_bias");
  check_bf16(norm_v_q_weight, "norm_v_q_weight");
  check_bf16(norm_v_k_weight, "norm_v_k_weight");
  check_bf16(packed_a, "packed_a");
  check_bf16(norm_a_q_weight, "norm_a_q_weight");
  check_bf16(norm_a_k_weight, "norm_a_k_weight");
  check_bf16(packed_u, "packed_u");
  check_bf16(norm_u_q_weight, "norm_u_q_weight");
  check_bf16(norm_u_k_weight, "norm_u_k_weight");
  const int heads = checked_int(heads64, "heads");
  const int head_dim = checked_int(head_dim64, "head_dim");
  check_qkv_common(packed_v, norm_v_q_weight, norm_v_k_weight, heads, head_dim, "packed_v");
  check_qkv_common(packed_a, norm_a_q_weight, norm_a_k_weight, heads, head_dim, "packed_a");
  check_qkv_common(packed_u, norm_u_q_weight, norm_u_k_weight, heads, head_dim, "packed_u");
  const int64_t batch = packed_v.size(0);
  TORCH_CHECK(batch == 1, "qkv_split_joint3_cat_bf16 currently supports batch == 1");
  TORCH_CHECK(packed_a.size(0) == batch && packed_u.size(0) == batch,
              "packed_v, packed_a, and packed_u must have the same batch");
  const int64_t video_seq_len = packed_v.size(1);
  const int64_t action_seq_len = packed_a.size(1);
  const int64_t und_seq_len = packed_u.size(1);
  const int64_t total_seq_len = video_seq_len + action_seq_len + und_seq_len;
  const int64_t dim = static_cast<int64_t>(heads) * head_dim;
  TORCH_CHECK(qkv_v_bias.dim() == 1 && qkv_v_bias.size(0) == 3 * dim,
              "qkv_v_bias must have shape (3 * heads * head_dim,)");
  TORCH_CHECK(rope_seq_len64 >= 0 && rope_seq_len64 <= video_seq_len,
              "rope_seq_len must be in [0, packed_v.shape[1]]");
  check_freqs(freqs_re, freqs_im, head_dim, rope_seq_len64);
  check_cat_out(q_cat_out, batch, total_seq_len, heads, head_dim, "q_cat_out");
  check_cat_out(k_cat_out, batch, total_seq_len, heads, head_dim, "k_cat_out");
  check_cat_out(v_cat_out, batch, total_seq_len, heads, head_dim, "v_cat_out");

  check_same_device(packed_v, qkv_v_bias, "packed_v", "qkv_v_bias");
  check_same_device(packed_v, norm_v_q_weight, "packed_v", "norm_v_q_weight");
  check_same_device(packed_v, norm_v_k_weight, "packed_v", "norm_v_k_weight");
  check_same_device(packed_v, freqs_re, "packed_v", "freqs_re");
  check_same_device(packed_v, freqs_im, "packed_v", "freqs_im");
  check_same_device(packed_v, packed_a, "packed_v", "packed_a");
  check_same_device(packed_v, packed_u, "packed_v", "packed_u");
  check_same_device(packed_v, q_cat_out, "packed_v", "q_cat_out");
  check_same_device(packed_v, k_cat_out, "packed_v", "k_cat_out");
  check_same_device(packed_v, v_cat_out, "packed_v", "v_cat_out");

#if defined(CUDA_KERNEL)
  at::cuda::CUDAGuard device_guard(packed_v.device());
  auto stream = at::cuda::getCurrentCUDAStream(packed_v.get_device()).stream();
  flash_rt::qkv_cache_rope::qkv_split_joint3_cat_bf16(
      packed_v.data_ptr(),
      qkv_v_bias.data_ptr(),
      norm_v_q_weight.data_ptr(),
      norm_v_k_weight.data_ptr(),
      reinterpret_cast<const float*>(freqs_re.data_ptr()),
      reinterpret_cast<const float*>(freqs_im.data_ptr()),
      packed_a.data_ptr(),
      norm_a_q_weight.data_ptr(),
      norm_a_k_weight.data_ptr(),
      packed_u.data_ptr(),
      norm_u_q_weight.data_ptr(),
      norm_u_k_weight.data_ptr(),
      q_cat_out.data_ptr(),
      k_cat_out.data_ptr(),
      v_cat_out.data_ptr(),
      checked_int(batch, "batch"),
      checked_int(total_seq_len, "total_seq_len"),
      checked_int(video_seq_len, "video_seq_len"),
      checked_int(action_seq_len, "action_seq_len"),
      checked_int(und_seq_len, "und_seq_len"),
      heads,
      head_dim,
      checked_int(rope_seq_len64, "rope_seq_len"),
      static_cast<float>(eps_v),
      static_cast<float>(eps_a),
      static_cast<float>(eps_u),
      stream);
#else
  TORCH_CHECK(false, "flashrt-qkv-cache-rope was not built with CUDA support");
#endif
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("decode_q_norm_rope_stage_bf16("
          "Tensor q_pre, Tensor q_norm_weight, Tensor cos, Tensor sin, "
          "float eps, Tensor! q_out) -> ()");
  ops.def("decode_k_norm_rope_kvwrite_bf16("
          "Tensor k_pre, Tensor v_pre, Tensor k_norm_weight, Tensor cos, Tensor sin, "
          "float eps, Tensor! k_cache_dst, Tensor! v_cache_dst) -> ()");
  ops.def("decode_k_norm_rope_kvwrite_devpos_bf16("
          "Tensor k_pre, Tensor v_pre, Tensor k_norm_weight, Tensor cos, Tensor sin, "
          "Tensor cur_pos, float eps, Tensor! k_cache, Tensor! v_cache) -> ()");
  ops.def("qkv_split_per_head_norm_rope_bf16("
          "Tensor packed_qkv, Tensor q_norm_weight, Tensor k_norm_weight, "
          "Tensor cos, Tensor sin, int q_heads, int kv_heads, float eps, "
          "Tensor! q_out, Tensor! k_out, Tensor! v_out) -> ()");
  ops.def("qkv_split_bias_rope_bf16("
          "Tensor packed_qkv, Tensor qkv_bias, Tensor cos, Tensor sin, "
          "int q_heads, int kv_heads, int head_dim, "
          "Tensor! q_out, Tensor! k_out, Tensor! v_out) -> ()");
  ops.def("qkv_split_bias_rope_fp16("
          "Tensor packed_qkv, Tensor qkv_bias, Tensor cos, Tensor sin, "
          "int q_heads, int kv_heads, int head_dim, "
          "Tensor! q_out, Tensor! k_out, Tensor! v_out) -> ()");
  ops.def("qkv_split_rope_kvcache_bf16("
          "Tensor packed_qkv, Tensor rope, int q_heads, int kv_heads, int head_dim, "
          "int cache_offset, Tensor! q_out, Tensor! k_cache, Tensor! v_cache) -> ()");
  ops.def("qkv_split_rope_kvcache_fp16("
          "Tensor packed_qkv, Tensor rope, int q_heads, int kv_heads, int head_dim, "
          "int cache_offset, Tensor? device_position, Tensor! q_out, "
          "Tensor! k_cache, Tensor! v_cache) -> ()");
  ops.def("qk_norm_rope_strided_bf16("
          "Tensor q_in, Tensor k_in, Tensor q_weight, Tensor k_weight, "
          "Tensor cos, Tensor sin, int q_heads, int k_heads, float eps, "
          "Tensor! q_out, Tensor! k_out) -> ()");
  ops.def("qkv_split_bf16("
          "Tensor packed_qkv, int heads, int head_dim, "
          "Tensor! q_out, Tensor! k_out, Tensor! v_out) -> ()");
  ops.def("qkv_split_norm_rope_bf16("
          "Tensor packed_qkv, Tensor norm_q_weight, Tensor norm_k_weight, "
          "Tensor freqs_re, Tensor freqs_im, int heads, int head_dim, "
          "int rope_seq_len, float eps, Tensor! q_out, Tensor! k_out) -> ()");
  ops.def("qkv_split_bias_norm_rope_v_bf16("
          "Tensor packed_qkv, Tensor qkv_bias, Tensor norm_q_weight, Tensor norm_k_weight, "
          "Tensor freqs_re, Tensor freqs_im, int heads, int head_dim, "
          "int rope_seq_len, float eps, Tensor! q_out, Tensor! k_out, Tensor! v_out) -> ()");
  ops.def("qkv_split_bias_norm_rope_v_cat_bf16("
          "Tensor packed_qkv, Tensor qkv_bias, Tensor norm_q_weight, Tensor norm_k_weight, "
          "Tensor freqs_re, Tensor freqs_im, int heads, int head_dim, int video_offset, "
          "int rope_seq_len, float eps, Tensor! q_cat_out, Tensor! k_cat_out, Tensor! v_cat_out) -> ()");
  ops.def("qkv_split_joint3_cat_bf16("
          "Tensor packed_v, Tensor qkv_v_bias, Tensor norm_v_q_weight, Tensor norm_v_k_weight, "
          "Tensor freqs_re, Tensor freqs_im, Tensor packed_a, Tensor norm_a_q_weight, "
          "Tensor norm_a_k_weight, Tensor packed_u, Tensor norm_u_q_weight, Tensor norm_u_k_weight, "
          "int heads, int head_dim, int rope_seq_len, float eps_v, float eps_a, float eps_u, "
          "Tensor! q_cat_out, Tensor! k_cat_out, Tensor! v_cat_out) -> ()");
#if defined(CUDA_KERNEL)
  ops.impl("decode_q_norm_rope_stage_bf16",
           torch::kCUDA,
           &decode_q_norm_rope_stage_bf16);
  ops.impl("decode_k_norm_rope_kvwrite_bf16",
           torch::kCUDA,
           &decode_k_norm_rope_kvwrite_bf16);
  ops.impl("decode_k_norm_rope_kvwrite_devpos_bf16",
           torch::kCUDA,
           &decode_k_norm_rope_kvwrite_devpos_bf16);
  ops.impl("qkv_split_per_head_norm_rope_bf16",
           torch::kCUDA,
           &qkv_split_per_head_norm_rope_bf16);
  ops.impl("qkv_split_bias_rope_bf16",
           torch::kCUDA,
           &qkv_split_bias_rope_bf16);
  ops.impl("qkv_split_bias_rope_fp16",
           torch::kCUDA,
           &qkv_split_bias_rope_fp16);
  ops.impl("qkv_split_rope_kvcache_bf16",
           torch::kCUDA,
           &qkv_split_rope_kvcache_bf16);
  ops.impl("qkv_split_rope_kvcache_fp16",
           torch::kCUDA,
           &qkv_split_rope_kvcache_fp16);
  ops.impl("qk_norm_rope_strided_bf16",
           torch::kCUDA,
           &qk_norm_rope_strided_bf16);
  ops.impl("qkv_split_bf16",
           torch::kCUDA,
           &qkv_split_bf16);
  ops.impl("qkv_split_norm_rope_bf16",
           torch::kCUDA,
           &qkv_split_norm_rope_bf16);
  ops.impl("qkv_split_bias_norm_rope_v_bf16",
           torch::kCUDA,
           &qkv_split_bias_norm_rope_v_bf16);
  ops.impl("qkv_split_bias_norm_rope_v_cat_bf16",
           torch::kCUDA,
           &qkv_split_bias_norm_rope_v_cat_bf16);
  ops.impl("qkv_split_joint3_cat_bf16",
           torch::kCUDA,
           &qkv_split_joint3_cat_bf16);
#elif defined(ROCM_KERNEL)
  ops.impl("qkv_split_rope_kvcache_bf16",
           torch::kCUDA,
           &qkv_split_rope_kvcache_bf16);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
