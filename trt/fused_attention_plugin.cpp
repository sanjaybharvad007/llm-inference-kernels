#include "fused_attention_plugin.h"

#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#include "flash_attn.h"

using namespace nvinfer1;

namespace fused_attn {

namespace {
ILoggerFinder* gLoggerFinder = nullptr;
void logError(char const* msg) {
  if (gLoggerFinder) if (auto* l = gLoggerFinder->findLogger()) l->log(ILogger::Severity::kERROR, msg);
}
}  // namespace

FusedAttentionPlugin::FusedAttentionPlugin(bool causal, float scale, int backend) : mCausal(causal), mScale(scale), mBackend(backend) {
  mFields.emplace_back(PluginField{"causal", nullptr, PluginFieldType::kINT32, 1});
  mFields.emplace_back(PluginField{"scale", nullptr, PluginFieldType::kFLOAT32, 1});
  mFields.emplace_back(PluginField{"backend", nullptr, PluginFieldType::kINT32, 1});
}

IPluginCapability* FusedAttentionPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept {
  switch (type) {
    case PluginCapabilityType::kBUILD:   return static_cast<IPluginV3OneBuild*>(this);
    case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
    case PluginCapabilityType::kCORE:    return static_cast<IPluginV3OneCore*>(this);
  }
  return nullptr;
}

IPluginV3* FusedAttentionPlugin::clone() noexcept {
  auto* p = new FusedAttentionPlugin(mCausal, mScale, mBackend);
  p->setPluginNamespace(mNamespace.c_str());
  return p;
}

int32_t FusedAttentionPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const*, int32_t) noexcept {
  if (nbInputs != 3) return -1;
  return 0;
}

bool FusedAttentionPlugin::supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t) noexcept {
  const auto& d = inOut[pos].desc;
  if (d.format != TensorFormat::kLINEAR) return false;
  const bool okType = d.type == DataType::kHALF || d.type == DataType::kBF16;
  // all tensors share one dtype
  return okType && d.type == inOut[0].desc.type;
}

int32_t FusedAttentionPlugin::getOutputDataTypes(DataType* outputTypes, int32_t, DataType const* inputTypes, int32_t) const noexcept {
  outputTypes[0] = inputTypes[0];
  return 0;
}

int32_t FusedAttentionPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t, DimsExprs* outputs, int32_t, IExprBuilder&) noexcept {
  if (nbInputs != 3 || inputs[0].nbDims != 4) return -1;
  outputs[0] = inputs[0];  // O has Q's shape
  return 0;
}

int32_t FusedAttentionPlugin::onShapeChange(PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const*, int32_t) noexcept {
  if (nbInputs != 3) return -1;
  const int d = in[0].dims.d[3];
  if (d != 64 && d != 128) { logError("FusedAttention: head_dim must be 64 or 128"); return -1; }
  if (in[0].dims.d[1] % in[1].dims.d[1] != 0) { logError("FusedAttention: H_q must be a multiple of H_kv"); return -1; }
  return 0;
}

int32_t FusedAttentionPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const*, void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept {
  const auto& q = inputDesc[0].dims; const auto& k = inputDesc[1].dims;
  AttnParams p{};
  p.q = inputs[0]; p.k = inputs[1]; p.v = inputs[2]; p.o = outputs[0];
  p.batch = q.d[0]; p.num_heads_q = q.d[1]; p.seqlen_q = q.d[2]; p.head_dim = q.d[3];
  p.num_heads_kv = k.d[1]; p.seqlen_k = k.d[2];
  // linear (row-major) layout: strides in elements
  p.q_row_stride = p.head_dim; p.q_head_stride = (int64_t)p.seqlen_q * p.head_dim; p.q_batch_stride = p.q_head_stride * p.num_heads_q;
  p.k_row_stride = p.head_dim; p.k_head_stride = (int64_t)p.seqlen_k * p.head_dim; p.k_batch_stride = p.k_head_stride * p.num_heads_kv;
  p.v_row_stride = p.k_row_stride; p.v_head_stride = p.k_head_stride; p.v_batch_stride = p.k_batch_stride;
  p.o_row_stride = p.q_row_stride; p.o_head_stride = p.q_head_stride; p.o_batch_stride = p.q_batch_stride;
  p.scale = mScale > 0.f ? mScale : 1.f / std::sqrt((float)p.head_dim);
  p.causal = mCausal;
  p.dtype = inputDesc[0].type == DataType::kHALF ? DType::kFloat16 : DType::kBFloat16;

  int dev = 0; cudaDeviceProp prop{};
  cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);
  const bool mma_ok = prop.major >= 8 && attention_fwd_mma_smem_bytes(p.head_dim, 2) <= (size_t)prop.sharedMemPerBlockOptin;
  if (mBackend == 2 || (mBackend == 0 && mma_ok)) run_attention_fwd_mma(p, stream);
  else run_attention_fwd(p, stream);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { logError(cudaGetErrorString(err)); return -1; }
  return 0;
}

PluginFieldCollection const* FusedAttentionPlugin::getFieldsToSerialize() noexcept {
  static int32_t causal; static float scale; static int32_t backend;
  causal = mCausal; scale = mScale; backend = mBackend;
  mFields[0].data = &causal; mFields[1].data = &scale; mFields[2].data = &backend;
  mFC.nbFields = (int32_t)mFields.size(); mFC.fields = mFields.data();
  return &mFC;
}

// ---- creator ---------------------------------------------------------------
FusedAttentionPluginCreator::FusedAttentionPluginCreator() {
  mFields.emplace_back(PluginField{"causal", nullptr, PluginFieldType::kINT32, 1});
  mFields.emplace_back(PluginField{"scale", nullptr, PluginFieldType::kFLOAT32, 1});
  mFields.emplace_back(PluginField{"backend", nullptr, PluginFieldType::kINT32, 1});
  mFC.nbFields = (int32_t)mFields.size(); mFC.fields = mFields.data();
}

IPluginV3* FusedAttentionPluginCreator::createPlugin(AsciiChar const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept {
  bool causal = false; float scale = 0.f; int backend = 0;
  for (int i = 0; fc && i < fc->nbFields; ++i) {
    const auto& f = fc->fields[i];
    if (!std::strcmp(f.name, "causal")) causal = *static_cast<int32_t const*>(f.data) != 0;
    else if (!std::strcmp(f.name, "scale")) scale = *static_cast<float const*>(f.data);
    else if (!std::strcmp(f.name, "backend")) backend = *static_cast<int32_t const*>(f.data);
  }
  auto* p = new FusedAttentionPlugin(causal, scale, backend);
  p->setPluginNamespace(mNamespace.c_str());
  return p;
}

}  // namespace fused_attn

extern "C" {
IPluginCreatorInterface* const* getCreators(int32_t& nbCreators) {
  static fused_attn::FusedAttentionPluginCreator creator;
  static IPluginCreatorInterface* creators[] = {&creator};
  nbCreators = 1;
  return creators;
}
void setLoggerFinder(ILoggerFinder* finder) { fused_attn::gLoggerFinder = finder; }
}
