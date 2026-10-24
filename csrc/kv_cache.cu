// KV-cache kernels: INT8 quantisation with per-token scales, paged append, and
// a decode attention kernel that reads the quantised cache directly.
//
// Decode is bandwidth-bound: one query row per head against the whole
// context.  Storing K/V as int8 halves the bytes streamed per step compared
// with fp16; dequantisation is fused into the dot product (one FMA per
// element plus one multiply per token row), and the accumulation stays in
// fp32, so the mixed-precision path costs no extra passes over memory.
//
// Layouts
//   dense  K/V  [B, H_kv, N, D]           (row-major, D contiguous)
//   paged  cache [num_blocks, block_size, H_kv, D] int8
//   scales       [num_blocks, block_size, H_kv]    fp32  (one per token per head)
//   block_tables [B, max_blocks] int32, context_lens [B] int32
//   q            [B, H_q, D]  fp16/bf16/fp32  (one decode step)
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "kv_cache.h"
#include "launch.h"

namespace fused_attn {
namespace {

constexpr int WARP = 32;

template <typename T> struct Cvt;
template <> struct Cvt<__half> {
  __device__ static float to_f32(__half x) { return __half2float(x); }
  __device__ static __half from_f32(float x) { return __float2half(x); }
};
template <> struct Cvt<__nv_bfloat16> {
  __device__ static float to_f32(__nv_bfloat16 x) { return __bfloat162float(x); }
  __device__ static __nv_bfloat16 from_f32(float x) { return __float2bfloat16(x); }
};
template <> struct Cvt<float> {
  __device__ static float to_f32(float x) { return x; }
  __device__ static float from_f32(float x) { return x; }
};

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ int8_t quantize(float x, float inv_scale) {
  const float r = rintf(x * inv_scale);
  return static_cast<int8_t>(fminf(127.f, fmaxf(-127.f, r)));
}

// ---------------------------------------------------------------------------
// 1. Quantise one token row (HEAD_DIM values) per warp: symmetric int8,
//    scale = max|x| / 127.  One warp per (token, head); lanes stride over D.
//    Used both for dense tensors and inside the paged append.
// ---------------------------------------------------------------------------
template <typename T, int HEAD_DIM>
__device__ __forceinline__ void quantize_row(const T* __restrict__ src, int8_t* __restrict__ dst, float* __restrict__ scale_out, int lane) {
  constexpr int PER_LANE = HEAD_DIM / WARP;
  float vals[PER_LANE];
  float amax = 0.f;
#pragma unroll
  for (int i = 0; i < PER_LANE; ++i) {
    vals[i] = Cvt<T>::to_f32(src[lane + i * WARP]);  // coalesced: consecutive lanes read consecutive elements
    amax = fmaxf(amax, fabsf(vals[i]));
  }
  amax = warp_max(amax);
  const float scale = fmaxf(amax, 1e-8f) / 127.f;
  const float inv = 1.f / scale;
#pragma unroll
  for (int i = 0; i < PER_LANE; ++i) dst[lane + i * WARP] = quantize(vals[i], inv);
  if (lane == 0) *scale_out = scale;
}

template <typename T, int HEAD_DIM>
__global__ void __launch_bounds__(128) quantize_kv_kernel(const T* __restrict__ x, int8_t* __restrict__ xq, float* __restrict__ scales,
                                                          int64_t rows, int64_t row_stride) {
  const int64_t row = (int64_t)blockIdx.x * 4 + threadIdx.x / WARP;  // 4 warps per block, one row each
  if (row >= rows) return;
  quantize_row<T, HEAD_DIM>(x + row * row_stride, xq + row * HEAD_DIM, scales + row, threadIdx.x % WARP);
}

// ---------------------------------------------------------------------------
// 2. Paged append: quantise the new tokens' K and V and scatter them into the
//    paged cache at slot_mapping[t] (= block * block_size + offset).
//    grid = (num_tokens, H_kv), block = 64 threads (one warp K, one warp V).
// ---------------------------------------------------------------------------
template <typename T, int HEAD_DIM>
__global__ void __launch_bounds__(64) paged_append_kernel(const T* __restrict__ k, const T* __restrict__ v,
                                                          int8_t* __restrict__ k_cache, int8_t* __restrict__ v_cache,
                                                          float* __restrict__ k_scale, float* __restrict__ v_scale,
                                                          const int32_t* __restrict__ slot_mapping,
                                                          int64_t tok_stride, int64_t head_stride, int H_kv) {
  const int t = blockIdx.x, h = blockIdx.y;
  const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
  const int64_t slot = slot_mapping[t];
  if (slot < 0) return;  // padding token
  const int64_t dst_row = slot * H_kv + h;            // row index into [num_blocks*block_size, H_kv]
  const T* src = (warp == 0 ? k : v) + (int64_t)t * tok_stride + (int64_t)h * head_stride;
  int8_t* dst = (warp == 0 ? k_cache : v_cache) + dst_row * HEAD_DIM;
  float* sc = (warp == 0 ? k_scale : v_scale) + dst_row;
  quantize_row<T, HEAD_DIM>(src, dst, sc, lane);
}

// ---------------------------------------------------------------------------
// 3. Decode attention over the int8 paged cache, split-K.
//    grid = (num_splits, H_q, B); block = 128 threads = 4 warps.
//    Each warp owns a run of key positions within the split; a lane handles
//    HEAD_DIM/32 elements of every key row (coalesced int8 loads: 4 bytes per
//    lane per row with HEAD_DIM=128).  Per split we produce an unnormalised
//    output, the running max and the running sum; a combine kernel merges
//    splits with the standard log-sum-exp rescale.
// ---------------------------------------------------------------------------
template <typename T, int HEAD_DIM, int BLOCK_SIZE>
__global__ void __launch_bounds__(128) decode_kv8_kernel(
    const T* __restrict__ q, const int8_t* __restrict__ k_cache, const int8_t* __restrict__ v_cache,
    const float* __restrict__ k_scale, const float* __restrict__ v_scale,
    const int32_t* __restrict__ block_tables, const int32_t* __restrict__ context_lens,
    float* __restrict__ o_partial,   // [B, H_q, splits, HEAD_DIM]
    float* __restrict__ m_partial,   // [B, H_q, splits]
    float* __restrict__ l_partial,   // [B, H_q, splits]
    int H_q, int H_kv, int max_blocks, int num_splits, float sm_scale) {
  constexpr int NWARPS = 4;
  constexpr int PER_LANE = HEAD_DIM / WARP;  // 4 for D=128, 2 for D=64
  const int split = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int h_kv = h / (H_q / H_kv);
  const int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
  const int ctx = context_lens[b];
  const int per_split = (ctx + num_splits - 1) / num_splits;
  const int begin = split * per_split;
  const int end = min(ctx, begin + per_split);

  __shared__ float red_m[NWARPS], red_l[NWARPS];
  __shared__ float red_o[NWARPS][HEAD_DIM];

  // q fragment, pre-scaled with log2(e) so exp2 can be used
  float qf[PER_LANE];
  const T* qp = q + ((int64_t)b * H_q + h) * HEAD_DIM;
  const float qk_scale = sm_scale * 1.4426950408889634f;
#pragma unroll
  for (int i = 0; i < PER_LANE; ++i) qf[i] = Cvt<T>::to_f32(qp[lane + i * WARP]) * qk_scale;

  float m_i = -CUDART_INF_F, l_i = 0.f;
  float acc[PER_LANE];
#pragma unroll
  for (int i = 0; i < PER_LANE; ++i) acc[i] = 0.f;

  const int32_t* bt = block_tables + (int64_t)b * max_blocks;
  // warps interleave over key positions: warp w takes pos = begin + w, begin + w + 4, ...
  for (int pos = begin + warp; pos < end; pos += NWARPS) {
    const int64_t row = ((int64_t)bt[pos / BLOCK_SIZE] * BLOCK_SIZE + (pos % BLOCK_SIZE)) * H_kv + h_kv;
    const int8_t* kr = k_cache + row * HEAD_DIM;
    // dot(q, dequant(k)) = k_scale * sum(q_i * kq_i): apply the scale once per row
    float dot = 0.f;
#pragma unroll
    for (int i = 0; i < PER_LANE; ++i) dot = fmaf(qf[i], (float)kr[lane + i * WARP], dot);
    dot = warp_sum(dot) * k_scale[row];
    const float m_new = fmaxf(m_i, dot);
    const float alpha = exp2f(m_i - m_new);
    const float p = exp2f(dot - m_new);
    l_i = l_i * alpha + p;
    const float pv = p * v_scale[row];
    const int8_t* vr = v_cache + row * HEAD_DIM;
#pragma unroll
    for (int i = 0; i < PER_LANE; ++i) acc[i] = fmaf(pv, (float)vr[lane + i * WARP], acc[i] * alpha);
    m_i = m_new;
  }

  // combine the 4 warps' partial (m, l, acc) in shared memory
  if (lane == 0) { red_m[warp] = m_i; red_l[warp] = l_i; }
#pragma unroll
  for (int i = 0; i < PER_LANE; ++i) red_o[warp][lane + i * WARP] = acc[i];
  __syncthreads();
  if (warp == 0) {
    float m = -CUDART_INF_F;
#pragma unroll
    for (int w = 0; w < NWARPS; ++w) m = fmaxf(m, red_m[w]);
    const float m_safe = (m == -CUDART_INF_F) ? 0.f : m;
    float l = 0.f;
    float o[PER_LANE];
#pragma unroll
    for (int i = 0; i < PER_LANE; ++i) o[i] = 0.f;
#pragma unroll
    for (int w = 0; w < NWARPS; ++w) {
      const float s = exp2f(red_m[w] - m_safe);
      l += red_l[w] * s;
#pragma unroll
      for (int i = 0; i < PER_LANE; ++i) o[i] += red_o[w][lane + i * WARP] * s;
    }
    const int64_t pidx = ((int64_t)b * H_q + h) * num_splits + split;
#pragma unroll
    for (int i = 0; i < PER_LANE; ++i) o_partial[pidx * HEAD_DIM + lane + i * WARP] = o[i];
    if (lane == 0) { m_partial[pidx] = m; l_partial[pidx] = l; }
  }
}

template <typename T, int HEAD_DIM>
__global__ void __launch_bounds__(128) combine_splits_kernel(const float* __restrict__ o_partial, const float* __restrict__ m_partial,
                                                             const float* __restrict__ l_partial, T* __restrict__ out,
                                                             int num_splits) {
  const int bh = blockIdx.x;  // b * H_q + h
  const int d = threadIdx.x;  // HEAD_DIM threads (<=128)
  if (d >= HEAD_DIM) return;
  float m = -CUDART_INF_F;
  for (int s = 0; s < num_splits; ++s) m = fmaxf(m, m_partial[(int64_t)bh * num_splits + s]);
  const float m_safe = (m == -CUDART_INF_F) ? 0.f : m;
  float l = 0.f, o = 0.f;
  for (int s = 0; s < num_splits; ++s) {
    const int64_t p = (int64_t)bh * num_splits + s;
    const float w = exp2f(m_partial[p] - m_safe);
    l += l_partial[p] * w;
    o += o_partial[p * HEAD_DIM + d] * w;
  }
  out[(int64_t)bh * HEAD_DIM + d] = Cvt<T>::from_f32(l > 0.f ? o / l : 0.f);
}

// ---- host dispatch ---------------------------------------------------------
template <typename T, int HEAD_DIM>
void quantize_kv_t(const KvQuantParams& p, cudaStream_t s) {
  const int64_t blocks = (p.rows + 3) / 4;
  auto kfn = &quantize_kv_kernel<T, HEAD_DIM>;
  KERNEL_LAUNCH(kfn, dim3((unsigned)blocks), dim3(128), 0, s)(static_cast<const T*>(p.x), static_cast<int8_t*>(p.xq), p.scales, p.rows, p.row_stride);
}

template <typename T, int HEAD_DIM>
void paged_append_t(const KvAppendParams& p, cudaStream_t s) {
  dim3 grid(p.num_tokens, p.num_heads_kv);
  auto kfn = &paged_append_kernel<T, HEAD_DIM>;
  KERNEL_LAUNCH(kfn, grid, dim3(64), 0, s)(static_cast<const T*>(p.k), static_cast<const T*>(p.v), p.k_cache, p.v_cache,
                                           p.k_scale, p.v_scale, p.slot_mapping, p.tok_stride, p.head_stride, p.num_heads_kv);
}

template <typename T, int HEAD_DIM, int BLOCK_SIZE>
void decode_kv8_t(const DecodeKv8Params& p, cudaStream_t s) {
  dim3 grid(p.num_splits, p.num_heads_q, p.batch);
  auto kdec = &decode_kv8_kernel<T, HEAD_DIM, BLOCK_SIZE>;
  KERNEL_LAUNCH(kdec, grid, dim3(128), 0, s)(static_cast<const T*>(p.q), p.k_cache, p.v_cache, p.k_scale, p.v_scale,
                                             p.block_tables, p.context_lens, p.o_partial, p.m_partial, p.l_partial,
                                             p.num_heads_q, p.num_heads_kv, p.max_blocks, p.num_splits, p.scale);
  auto kcmb = &combine_splits_kernel<T, HEAD_DIM>;
  KERNEL_LAUNCH(kcmb, dim3(p.batch * p.num_heads_q), dim3(128), 0, s)(p.o_partial, p.m_partial, p.l_partial, static_cast<T*>(p.out), p.num_splits);
}

#define DISPATCH_T_D(FN, P, S)                                                                   \
  switch ((P).dtype) {                                                                           \
    case DType::kFloat16:  if ((P).head_dim == 64) FN<__half, 64>(P, S); else FN<__half, 128>(P, S); break;          \
    case DType::kBFloat16: if ((P).head_dim == 64) FN<__nv_bfloat16, 64>(P, S); else FN<__nv_bfloat16, 128>(P, S); break; \
    case DType::kFloat32:  if ((P).head_dim == 64) FN<float, 64>(P, S); else FN<float, 128>(P, S); break;            \
  }

}  // namespace

void run_quantize_kv(const KvQuantParams& p, cudaStream_t s) { DISPATCH_T_D(quantize_kv_t, p, s) }
void run_paged_append(const KvAppendParams& p, cudaStream_t s) { DISPATCH_T_D(paged_append_t, p, s) }

void run_decode_kv8(const DecodeKv8Params& p, cudaStream_t s) {
#define DECODE_BS(BS)                                                                                              \
  switch (p.dtype) {                                                                                               \
    case DType::kFloat16:  if (p.head_dim == 64) decode_kv8_t<__half, 64, BS>(p, s); else decode_kv8_t<__half, 128, BS>(p, s); break;                 \
    case DType::kBFloat16: if (p.head_dim == 64) decode_kv8_t<__nv_bfloat16, 64, BS>(p, s); else decode_kv8_t<__nv_bfloat16, 128, BS>(p, s); break;   \
    case DType::kFloat32:  if (p.head_dim == 64) decode_kv8_t<float, 64, BS>(p, s); else decode_kv8_t<float, 128, BS>(p, s); break;                   \
  }
  switch (p.block_size) {
    case 16: DECODE_BS(16) break;
    case 32: DECODE_BS(32) break;
    default: DECODE_BS(64) break;
  }
#undef DECODE_BS
}

int decode_kv8_num_splits(int context_len, int batch_heads, int num_sms) {
  // enough blocks to fill the machine ~2x, each split at least 256 keys
  int splits = max(1, (num_sms * 2 + batch_heads - 1) / batch_heads);
  splits = min(splits, max(1, context_len / 256));
  return min(splits, 64);
}

}  // namespace fused_attn
