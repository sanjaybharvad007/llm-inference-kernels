// Host-side stand-ins so the kernels type-check with a plain C++ compiler in CI (no GPU, no nvcc):
//   g++ -fsyntax-only -std=c++17 -x c++ -Icsrc -Icsrc/host_stub -include csrc/host_stub/cuda_host_stub.h <file>.cu
// Nothing here is meant to run.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__
#define __shared__ static
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(...)
#define CUDART_INF_F (__builtin_inff())
#define FUSED_ATTN_HOST_STUB 1
using std::min; using std::max;
struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
static dim3 threadIdx, blockIdx, blockDim, gridDim;
typedef struct CUstream_st* cudaStream_t;
typedef int cudaError_t;
inline void __syncthreads() {}
inline float __shfl_xor_sync(unsigned, float v, int) { return v; }
inline int __shfl_xor_sync(unsigned, int v, int) { return v; }
inline float __shfl_sync(unsigned, float v, int) { return v; }
inline float exp2f_(float x) { return exp2f(x); }
inline float rintf_(float x) { return rintf(x); }
struct __half { unsigned short v; };
struct __nv_bfloat16 { unsigned short v; };
struct __half2 { __half x, y; };
struct __nv_bfloat162 { __nv_bfloat16 x, y; };
typedef __half half;
struct float2 { float x, y; }; struct float4 { float x, y, z, w; }; struct uint4 { unsigned x, y, z, w; }; struct uint2 { unsigned x, y; };
inline float __half2float(__half) { return 0.f; }
inline __half __float2half(float) { return __half{}; }
inline float __bfloat162float(__nv_bfloat16) { return 0.f; }
inline __nv_bfloat16 __float2bfloat16(float) { return __nv_bfloat16{}; }
inline float2 __half22float2(__half2) { return float2{}; }
inline float2 __bfloat1622float2(__nv_bfloat162) { return float2{}; }
inline __half2 __floats2half2_rn(float, float) { return __half2{}; }
inline __nv_bfloat162 __floats2bfloat162_rn(float, float) { return __nv_bfloat162{}; }
template <typename T> inline T* __ldg(const T* p) { return const_cast<T*>(p); }
// kernel launch syntax <<<grid, block, smem, stream>>> cannot be parsed by g++: files guard launches with FUSED_ATTN_HOST_STUB
