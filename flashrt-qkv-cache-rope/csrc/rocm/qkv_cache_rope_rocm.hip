// SPDX-License-Identifier: Apache-2.0

#include "rocm/qkv_cache_rope_rocm.h"

#include <cstdint>

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

namespace flash_rt {
namespace qkv_cache_rope {
namespace {

constexpr int kThreads = 256;

inline bool is_aligned_16(const void* ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 15u) == 0;
}

__device__ __forceinline__ float to_float(hip_bfloat16 value) {
  return static_cast<float>(value);
}

__device__ __forceinline__ hip_bfloat16 to_bf16(float value) {
  return hip_bfloat16(value);
}

__global__ void qkv_split_rope_kvcache_bf16_scalar_kernel(
    const hip_bfloat16* __restrict__ packed_qkv,
    const hip_bfloat16* __restrict__ rope,
    hip_bfloat16* __restrict__ q_out,
    hip_bfloat16* __restrict__ k_cache,
    hip_bfloat16* __restrict__ v_cache,
    int batch,
    int seq_len,
    int max_seq_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int cache_offset) {
  const int q_dim = q_heads * head_dim;
  const int kv_dim = kv_heads * head_dim;
  const int qkv_stride = q_dim + 2 * kv_dim;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * seq_len * qkv_stride;
  if (idx >= total) {
    return;
  }

  const int col = idx % qkv_stride;
  const int row = idx / qkv_stride;
  const int seq = row % seq_len;
  const int batch_idx = row / seq_len;
  if (col < q_dim) {
    const int dim = col % head_dim;
    const int pair = dim >> 1;
    const bool odd = (dim & 1) != 0;
    const int head = col / head_dim;
    const long long pair_base =
        (static_cast<long long>(batch_idx) * seq_len + seq) * qkv_stride +
        head * head_dim + pair * 2;
    const float x0 = to_float(packed_qkv[pair_base]);
    const float x1 = to_float(packed_qkv[pair_base + 1]);
    const long long rope_base =
        static_cast<long long>(seq) * head_dim + pair * 2;
    const float cos_value = to_float(rope[rope_base]);
    const float sin_value = to_float(rope[rope_base + 1]);
    const long long out_idx =
        (static_cast<long long>(batch_idx) * seq_len + seq) * q_dim + col;
    q_out[out_idx] = to_bf16(
        odd ? x1 * cos_value + x0 * sin_value
            : x0 * cos_value - x1 * sin_value);
  } else if (col < q_dim + kv_dim) {
    const int k_col = col - q_dim;
    const int dim = k_col % head_dim;
    const int pair = dim >> 1;
    const bool odd = (dim & 1) != 0;
    const int head = k_col / head_dim;
    const long long pair_base =
        (static_cast<long long>(batch_idx) * seq_len + seq) * qkv_stride +
        q_dim + head * head_dim + pair * 2;
    const float x0 = to_float(packed_qkv[pair_base]);
    const float x1 = to_float(packed_qkv[pair_base + 1]);
    const long long rope_base =
        static_cast<long long>(seq) * head_dim + pair * 2;
    const float cos_value = to_float(rope[rope_base]);
    const float sin_value = to_float(rope[rope_base + 1]);
    const long long cache_idx =
        (static_cast<long long>(batch_idx) * max_seq_len + cache_offset + seq) *
            kv_dim +
        k_col;
    k_cache[cache_idx] = to_bf16(
        odd ? x1 * cos_value + x0 * sin_value
            : x0 * cos_value - x1 * sin_value);
  } else {
    const int v_col = col - q_dim - kv_dim;
    const long long cache_idx =
        (static_cast<long long>(batch_idx) * max_seq_len + cache_offset + seq) *
            kv_dim +
        v_col;
    v_cache[cache_idx] = packed_qkv[idx];
  }
}

__device__ __forceinline__ int4 rotate_four_pairs(int4 x_raw, int4 rope_raw) {
  const auto* x = reinterpret_cast<const hip_bfloat16*>(&x_raw);
  const auto* rope = reinterpret_cast<const hip_bfloat16*>(&rope_raw);
  int4 out_raw;
  auto* out = reinterpret_cast<hip_bfloat16*>(&out_raw);
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const float x0 = to_float(x[2 * pair]);
    const float x1 = to_float(x[2 * pair + 1]);
    const float cos_value = to_float(rope[2 * pair]);
    const float sin_value = to_float(rope[2 * pair + 1]);
    out[2 * pair] = to_bf16(x0 * cos_value - x1 * sin_value);
    out[2 * pair + 1] = to_bf16(x1 * cos_value + x0 * sin_value);
  }
  return out_raw;
}

__global__ void qkv_split_rope_kvcache_bf16_vec_kernel(
    const int4* __restrict__ packed_qkv,
    const int4* __restrict__ rope,
    int4* __restrict__ q_out,
    int4* __restrict__ k_cache,
    int4* __restrict__ v_cache,
    int batch,
    int seq_len,
    int max_seq_len,
    int q_dim8,
    int kv_dim8,
    int head_dim8,
    int qkv_stride8,
    int cache_offset) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch * seq_len * qkv_stride8;
  if (idx >= total) {
    return;
  }

  const int col8 = idx % qkv_stride8;
  const int row = idx / qkv_stride8;
  const int seq = row % seq_len;
  const int batch_idx = row / seq_len;
  if (col8 < q_dim8) {
    q_out[(batch_idx * seq_len + seq) * q_dim8 + col8] =
        rotate_four_pairs(packed_qkv[idx],
                          rope[seq * head_dim8 + col8 % head_dim8]);
  } else if (col8 < q_dim8 + kv_dim8) {
    const int k_col8 = col8 - q_dim8;
    k_cache[((batch_idx * max_seq_len + cache_offset + seq) * kv_dim8) +
            k_col8] =
        rotate_four_pairs(packed_qkv[idx],
                          rope[seq * head_dim8 + k_col8 % head_dim8]);
  } else {
    const int v_col8 = col8 - q_dim8 - kv_dim8;
    v_cache[((batch_idx * max_seq_len + cache_offset + seq) * kv_dim8) +
            v_col8] = packed_qkv[idx];
  }
}

}  // namespace

void qkv_split_rope_kvcache_bf16_rocm(
    const void* packed_qkv,
    const void* rope,
    void* q_out,
    void* k_cache,
    void* v_cache,
    int batch,
    int seq_len,
    int max_seq_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int cache_offset,
    hipStream_t stream) {
  if (batch <= 0 || seq_len <= 0 || max_seq_len <= 0 || q_heads <= 0 ||
      kv_heads <= 0 || head_dim <= 0) {
    return;
  }

  if ((head_dim & 7) == 0 && is_aligned_16(packed_qkv) &&
      is_aligned_16(rope) && is_aligned_16(q_out) &&
      is_aligned_16(k_cache) && is_aligned_16(v_cache)) {
    const int q_dim8 = q_heads * (head_dim >> 3);
    const int kv_dim8 = kv_heads * (head_dim >> 3);
    const int qkv_stride8 = q_dim8 + 2 * kv_dim8;
    const int total = batch * seq_len * qkv_stride8;
    const int blocks = (total + kThreads - 1) / kThreads;
    hipLaunchKernelGGL(qkv_split_rope_kvcache_bf16_vec_kernel,
                       dim3(blocks), dim3(kThreads), 0, stream,
                       static_cast<const int4*>(packed_qkv),
                       static_cast<const int4*>(rope),
                       static_cast<int4*>(q_out),
                       static_cast<int4*>(k_cache),
                       static_cast<int4*>(v_cache),
                       batch, seq_len, max_seq_len, q_dim8, kv_dim8,
                       head_dim >> 3, qkv_stride8, cache_offset);
    return;
  }

  const int qkv_stride = (q_heads + 2 * kv_heads) * head_dim;
  const int total = batch * seq_len * qkv_stride;
  const int blocks = (total + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(qkv_split_rope_kvcache_bf16_scalar_kernel,
                     dim3(blocks), dim3(kThreads), 0, stream,
                     static_cast<const hip_bfloat16*>(packed_qkv),
                     static_cast<const hip_bfloat16*>(rope),
                     static_cast<hip_bfloat16*>(q_out),
                     static_cast<hip_bfloat16*>(k_cache),
                     static_cast<hip_bfloat16*>(v_cache),
                     batch, seq_len, max_seq_len, q_heads, kv_heads,
                     head_dim, cache_offset);
}

}  // namespace qkv_cache_rope
}  // namespace flash_rt
