// PyTorch bindings for the CUDA attention kernel.
#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "flash_attn.h"
#include "kv_cache.h"

namespace {

fused_attn::DType to_dtype(at::ScalarType t) {
  switch (t) {
    case at::kHalf: return fused_attn::DType::kFloat16;
    case at::kBFloat16: return fused_attn::DType::kBFloat16;
    case at::kFloat: return fused_attn::DType::kFloat32;
    default: TORCH_CHECK(false, "unsupported dtype ", t);
  }
}

fused_attn::AttnParams make_params(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                                   torch::Tensor& o, bool causal, double scale) {
  fused_attn::AttnParams p{};
  p.q = q.data_ptr(); p.k = k.data_ptr(); p.v = v.data_ptr(); p.o = o.data_ptr();
  p.q_batch_stride = q.stride(0); p.q_head_stride = q.stride(1); p.q_row_stride = q.stride(2);
  p.k_batch_stride = k.stride(0); p.k_head_stride = k.stride(1); p.k_row_stride = k.stride(2);
  p.v_batch_stride = v.stride(0); p.v_head_stride = v.stride(1); p.v_row_stride = v.stride(2);
  p.o_batch_stride = o.stride(0); p.o_head_stride = o.stride(1); p.o_row_stride = o.stride(2);
  p.batch = q.size(0); p.num_heads_q = q.size(1); p.num_heads_kv = k.size(1);
  p.seqlen_q = q.size(2); p.seqlen_k = k.size(2); p.head_dim = q.size(3);
  p.scale = static_cast<float>(scale);
  p.causal = causal;
  p.dtype = to_dtype(q.scalar_type());
  return p;
}

void check_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "expected [B, H, N, D]");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type());
  TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "head_dim must be contiguous");
  const int64_t d = q.size(3);
  TORCH_CHECK(d == 64 || d == 128, "CUDA kernel supports head_dim 64/128 (use the Triton path otherwise)");
  TORCH_CHECK(q.size(1) % k.size(1) == 0, "H_q must be a multiple of H_kv");
  TORCH_CHECK(k.sizes() == v.sizes());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0 &&
              reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "K/V must be 16B aligned");
  const int64_t vec = 16 / k.element_size();  // elements per 16B vector load
  TORCH_CHECK(k.stride(2) % vec == 0 && v.stride(2) % vec == 0 && k.stride(1) % vec == 0 && v.stride(1) % vec == 0,
              "K/V row and head strides must be multiples of ", vec, " elements for vectorised loads");

}

torch::Tensor attention_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                            bool causal, double scale, int64_t config) {
  check_inputs(q, k, v);
  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto o = torch::empty_like(q);
  auto p = make_params(q, k, v, o, causal, scale);
  p.config = static_cast<int>(config);
  fused_attn::run_attention_fwd(p, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}

torch::Tensor attention_fwd_mma(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
                                bool causal, double scale) {
  check_inputs(q, k, v);
  TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16, "Tensor-Core kernel needs fp16/bf16");
  TORCH_CHECK(q.stride(2) % 8 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0,
              "Q rows must be 16B aligned for cp.async");
  TORCH_CHECK(q.stride(2) % 2 == 0, "row strides must be even for 32-bit vector stores");
  const auto prop = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major >= 8, "Tensor-Core kernel requires sm_80+ (got sm_", prop->major, prop->minor, ")");
  const size_t smem = fused_attn::attention_fwd_mma_smem_bytes(q.size(3), q.element_size());
  TORCH_CHECK(smem <= static_cast<size_t>(prop->sharedMemPerBlockOptin), "not enough shared memory per block: need ",
              smem, " bytes");
  const at::cuda::OptionalCUDAGuard guard(q.device());
  auto o = torch::empty_like(q);
  auto p = make_params(q, k, v, o, causal, scale);
  fused_attn::run_attention_fwd_mma(p, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return o;
}

// ---- KV cache (int8) ----------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> quantize_kv(const torch::Tensor& x) {
  TORCH_CHECK(x.is_cuda() && x.dim() >= 2 && x.stride(-1) == 1, "x must be CUDA with contiguous last dim");
  const int64_t d = x.size(-1);
  TORCH_CHECK(d == 64 || d == 128, "head_dim 64/128");
  auto x2 = x.reshape({-1, d});
  TORCH_CHECK(x2.stride(1) == 1);
  const at::cuda::OptionalCUDAGuard guard(x.device());
  auto xq = torch::empty_like(x2, x2.options().dtype(torch::kInt8));
  auto scales = torch::empty({x2.size(0)}, x2.options().dtype(torch::kFloat32));
  fused_attn::KvQuantParams p{x2.data_ptr(), xq.data_ptr(), scales.data_ptr<float>(), x2.size(0), x2.stride(0), (int)d, to_dtype(x.scalar_type())};
  fused_attn::run_quantize_kv(p, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto sizes = x.sizes().vec(); sizes.pop_back();
  return {xq.view(x.sizes()), scales.view(sizes)};
}

void paged_append_kv8(const torch::Tensor& k, const torch::Tensor& v, torch::Tensor& k_cache, torch::Tensor& v_cache,
                      torch::Tensor& k_scale, torch::Tensor& v_scale, const torch::Tensor& slot_mapping) {
  // k, v: [T, H_kv, D]; caches: [num_blocks, block_size, H_kv, D] int8; scales: [num_blocks, block_size, H_kv] fp32
  TORCH_CHECK(k.is_cuda() && k.dim() == 3 && k.stride(2) == 1 && v.sizes() == k.sizes() && v.strides() == k.strides(), "k/v must be [T, H_kv, D] with contiguous D");
  TORCH_CHECK(k_cache.dtype() == torch::kInt8 && k_cache.is_contiguous() && v_cache.is_contiguous() && k_scale.is_contiguous() && v_scale.is_contiguous());
  TORCH_CHECK(slot_mapping.dtype() == torch::kInt32 && slot_mapping.numel() == k.size(0));
  const at::cuda::OptionalCUDAGuard guard(k.device());
  fused_attn::KvAppendParams p{k.data_ptr(), v.data_ptr(), k_cache.data_ptr<int8_t>(), v_cache.data_ptr<int8_t>(), k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
                               slot_mapping.data_ptr<int32_t>(), k.stride(0), k.stride(1), (int)k.size(0), (int)k.size(1), (int)k.size(2), to_dtype(k.scalar_type())};
  fused_attn::run_paged_append(p, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor decode_attention_kv8(const torch::Tensor& q, const torch::Tensor& k_cache, const torch::Tensor& v_cache,
                                   const torch::Tensor& k_scale, const torch::Tensor& v_scale,
                                   const torch::Tensor& block_tables, const torch::Tensor& context_lens, double scale, int64_t num_splits) {
  // q: [B, H_q, D]; caches: [num_blocks, block_size, H_kv, D] int8; scales [num_blocks, block_size, H_kv]
  TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.is_contiguous(), "q must be contiguous [B, H_q, D]");
  TORCH_CHECK(k_cache.dtype() == torch::kInt8 && k_cache.dim() == 4 && k_cache.is_contiguous() && v_cache.is_contiguous());
  TORCH_CHECK(block_tables.dtype() == torch::kInt32 && context_lens.dtype() == torch::kInt32);
  const int B = q.size(0), Hq = q.size(1), D = q.size(2), Hkv = k_cache.size(2), bs = k_cache.size(1);
  TORCH_CHECK(D == 64 || D == 128); TORCH_CHECK(bs == 16 || bs == 32 || bs == 64, "block_size 16/32/64");
  const at::cuda::OptionalCUDAGuard guard(q.device());
  const auto prop = at::cuda::getCurrentDeviceProperties();
  const int max_ctx = context_lens.max().item<int>();
  int splits = num_splits > 0 ? (int)num_splits : fused_attn::decode_kv8_num_splits(max_ctx, B * Hq, prop->multiProcessorCount);
  auto f32 = q.options().dtype(torch::kFloat32);
  auto o_part = torch::empty({B, Hq, splits, D}, f32), m_part = torch::empty({B, Hq, splits}, f32), l_part = torch::empty({B, Hq, splits}, f32);
  auto out = torch::empty_like(q);
  auto bt = block_tables.contiguous(), cl = context_lens.contiguous();
  fused_attn::DecodeKv8Params p{q.data_ptr(), k_cache.data_ptr<int8_t>(), v_cache.data_ptr<int8_t>(), k_scale.data_ptr<float>(), v_scale.data_ptr<float>(),
                                bt.data_ptr<int32_t>(), cl.data_ptr<int32_t>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), out.data_ptr(),
                                B, Hq, Hkv, D, bs, (int)bt.size(1), splits, (float)scale, to_dtype(q.scalar_type())};
  fused_attn::run_decode_kv8(p, at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<std::tuple<int, int, int, int, int>> simt_configs_py(int64_t head_dim, int64_t elem_size) {
  fused_attn::SimtConfig cfgs[8];
  const int n = fused_attn::simt_configs((int)head_dim, (int)elem_size, cfgs, 8);
  std::vector<std::tuple<int, int, int, int, int>> out;
  for (int i = 0; i < n && i < 8; ++i) out.emplace_back(cfgs[i].id, cfgs[i].block_m, cfgs[i].block_n, cfgs[i].threads, cfgs[i].smem_bytes);
  return out;
}

bool has_tensor_core_kernel() {
  if (!at::cuda::is_available()) return false;
  return at::cuda::getCurrentDeviceProperties()->major >= 8;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention_fwd", &attention_fwd, "Fused attention forward (CUDA, SIMT)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false, py::arg("scale") = 0.0, py::arg("config") = 0);
  m.def("simt_configs", &simt_configs_py, "Available SIMT tile configs: (id, block_m, block_n, threads, smem_bytes)",
        py::arg("head_dim"), py::arg("elem_size"));
  m.def("attention_fwd_mma", &attention_fwd_mma, "Fused attention forward (CUDA, Tensor Cores: mma.sync)",
        py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false, py::arg("scale") = 0.0);
  m.def("has_tensor_core_kernel", &has_tensor_core_kernel, "True on sm_80+ devices");
  m.def("quantize_kv", &quantize_kv, "Symmetric int8 quantisation with one scale per row (token, head)");
  m.def("paged_append_kv8", &paged_append_kv8, "Quantise new K/V tokens and scatter into the paged int8 cache");
  m.def("decode_attention_kv8", &decode_attention_kv8, "Split-K decode attention over the int8 paged cache",
        py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("k_scale"), py::arg("v_scale"), py::arg("block_tables"), py::arg("context_lens"),
        py::arg("scale"), py::arg("num_splits") = 0);
}
