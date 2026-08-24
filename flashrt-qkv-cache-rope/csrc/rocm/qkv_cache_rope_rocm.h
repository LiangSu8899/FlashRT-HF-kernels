// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <hip/hip_runtime.h>

namespace flash_rt {
namespace qkv_cache_rope {

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
    hipStream_t stream);

}  // namespace qkv_cache_rope
}  // namespace flash_rt
