#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "flash_attn.h"

namespace fused_attn {

struct KvQuantParams {
  const void* x;      // [rows, head_dim] rows of dtype
  void* xq;           // int8 [rows, head_dim]
  float* scales;      // [rows]
  int64_t rows, row_stride;
  int head_dim;
  DType dtype;
};

struct KvAppendParams {
  const void* k; const void* v;             // [num_tokens, H_kv, D] (strided)
  int8_t* k_cache; int8_t* v_cache;         // [num_blocks * block_size, H_kv, D]
  float* k_scale; float* v_scale;           // [num_blocks * block_size, H_kv]
  const int32_t* slot_mapping;              // [num_tokens], -1 = skip
  int64_t tok_stride, head_stride;
  int num_tokens, num_heads_kv, head_dim;
  DType dtype;
};

struct DecodeKv8Params {
  const void* q;                            // [B, H_q, D]
  const int8_t* k_cache; const int8_t* v_cache;
  const float* k_scale; const float* v_scale;
  const int32_t* block_tables;              // [B, max_blocks]
  const int32_t* context_lens;              // [B]
  float* o_partial; float* m_partial; float* l_partial;   // workspace [B,H_q,splits,D], [B,H_q,splits] x2
  void* out;                                // [B, H_q, D]
  int batch, num_heads_q, num_heads_kv, head_dim, block_size, max_blocks, num_splits;
  float scale;
  DType dtype;
};

void run_quantize_kv(const KvQuantParams& p, cudaStream_t s);
void run_paged_append(const KvAppendParams& p, cudaStream_t s);
void run_decode_kv8(const DecodeKv8Params& p, cudaStream_t s);
int decode_kv8_num_splits(int context_len, int batch_heads, int num_sms);

}  // namespace fused_attn
