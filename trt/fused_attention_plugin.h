// TensorRT plugin exposing the fused attention kernels (SIMT and Tensor-Core)
// as a single layer, so a TensorRT engine can replace the MatMul-Softmax-MatMul
// subgraph with one kernel launch.  Written against the TensorRT 10 IPluginV3
// interface (IPluginV3OneCore / OneBuild / OneRuntime).
//
// Inputs : Q [B, H_q, N_q, D], K [B, H_kv, N_k, D], V [B, H_kv, N_k, D]   (fp16 or bf16, linear format)
// Output : O [B, H_q, N_q, D]
// Attributes: causal (int32), scale (float32, 0 = 1/sqrt(D)), backend (int32: 0 auto, 1 simt, 2 mma)
#pragma once
#include <NvInferRuntime.h>
#include <string>
#include <vector>

namespace fused_attn {

class FusedAttentionPlugin : public nvinfer1::IPluginV3,
                             public nvinfer1::IPluginV3OneCore,
                             public nvinfer1::IPluginV3OneBuild,
                             public nvinfer1::IPluginV3OneRuntime {
 public:
  FusedAttentionPlugin(bool causal, float scale, int backend);

  // IPluginV3
  nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
  nvinfer1::IPluginV3* clone() noexcept override;

  // IPluginV3OneCore
  nvinfer1::AsciiChar const* getPluginName() const noexcept override { return kName; }
  nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return kVersion; }
  nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
  void setPluginNamespace(char const* ns) noexcept { mNamespace = ns; }

  // IPluginV3OneBuild
  int32_t getNbOutputs() const noexcept override { return 1; }
  int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs, nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
  bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
  int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
  int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, int32_t, nvinfer1::DynamicPluginTensorDesc const*, int32_t) const noexcept override { return 0; }

  // IPluginV3OneRuntime
  int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
  int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
  nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override { return clone(); }
  nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

  static constexpr char const* kName = "FusedAttention";
  static constexpr char const* kVersion = "1";

 private:
  bool mCausal;
  float mScale;
  int mBackend;
  std::string mNamespace;
  std::vector<nvinfer1::PluginField> mFields;
  nvinfer1::PluginFieldCollection mFC{};
};

class FusedAttentionPluginCreator : public nvinfer1::IPluginCreatorV3One {
 public:
  FusedAttentionPluginCreator();
  nvinfer1::AsciiChar const* getPluginName() const noexcept override { return FusedAttentionPlugin::kName; }
  nvinfer1::AsciiChar const* getPluginVersion() const noexcept override { return FusedAttentionPlugin::kVersion; }
  nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
  nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &mFC; }
  nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

 private:
  std::string mNamespace;
  std::vector<nvinfer1::PluginField> mFields;
  nvinfer1::PluginFieldCollection mFC{};
};

}  // namespace fused_attn

extern "C" {
// Entry points TensorRT looks for when loading the plugin library (trtexec --plugins / Python ctypes.CDLL).
nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& nbCreators);
void setLoggerFinder(nvinfer1::ILoggerFinder* finder);
}
