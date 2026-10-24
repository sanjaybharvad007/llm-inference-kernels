#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace fused_attn {

enum class DType { kFloat16, kBFloat16, kFloat32 };

struct AttnParams {
  const void* q;
  const void* k;
  const void* v;
  void* o;
  int64_t q_batch_stride, q_head_stride, q_row_stride;
  int64_t k_batch_stride, k_head_stride, k_row_stride;
  int64_t v_batch_stride, v_head_stride, v_row_stride;
  int64_t o_batch_stride, o_head_stride, o_row_stride;
  int batch, num_heads_q, num_heads_kv, seqlen_q, seqlen_k, head_dim;
  float scale;
  bool causal;
  DType dtype;
  /** SIMT kernel tile configuration id (0 = default). See simt_configs(). */
  int config = 0;
};

/** One launch configuration of the SIMT kernel: how the work is cut into blocks/warps/shared memory. */
struct SimtConfig {
  int id;
  int block_m;      // query rows per thread block
  int block_n;      // keys per shared-memory tile
  int threads;      // threads per block (= block_m * 2)
  int smem_bytes;   // static shared memory (double-buffered K and V tiles)
};
/** Configurations available for a head_dim / element size (static smem <= 48 KB, register budget respected). */
int simt_configs(int head_dim, int elem_size, SimtConfig* out, int max_out);

// SIMT kernel (fp16 / bf16 / fp32, head_dim 64/128, any arch)
void run_attention_fwd(const AttnParams& p, cudaStream_t stream);
// Tensor-Core kernel: mma.sync + ldmatrix + cp.async (fp16 / bf16, head_dim 64/128, sm_80+)
void run_attention_fwd_mma(const AttnParams& p, cudaStream_t stream);
size_t attention_fwd_mma_smem_bytes(int head_dim, int elem_size);

}  // namespace fused_attn
