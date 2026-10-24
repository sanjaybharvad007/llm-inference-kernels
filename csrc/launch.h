#pragma once
// Kernel launch helper that also compiles under the host stub (g++ cannot parse <<<>>>).
//   auto k = &my_kernel<T>;  KERNEL_LAUNCH(k, grid, block, smem, stream)(args...);
#ifdef FUSED_ATTN_HOST_STUB
#define KERNEL_LAUNCH(kernel, grid, block, smem, stream) if (false) (*(kernel))
#else
#define KERNEL_LAUNCH(kernel, grid, block, smem, stream) (*(kernel))<<<(grid), (block), (smem), (stream)>>>
#endif
