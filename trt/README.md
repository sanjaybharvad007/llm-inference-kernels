# TensorRT plugin

`libfused_attention_trt.so` exposes the fused attention kernels as a TensorRT
layer (`FusedAttention`, version `1`, IPluginV3 / TensorRT 10) so an engine can
run the whole attention block as one kernel instead of the MatMul → Softmax →
MatMul chain.

```bash
cmake -S trt -B trt/build -DTENSORRT_ROOT=/usr/lib/x86_64-linux-gnu -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
cmake --build trt/build -j
python trt/build_engine.py --plugin trt/build/libfused_attention_trt.so --batch 4 --seqlen 2048 --causal
trtexec --plugins=trt/build/libfused_attention_trt.so --onnx=model_with_plugin.onnx     # or from ONNX
```

Inputs `Q [B,H_q,N_q,D]`, `K/V [B,H_kv,N_k,D]` (fp16/bf16, linear), output `O`
with Q's shape. Attributes: `causal` (int32), `scale` (float32, 0 → 1/√D),
`backend` (0 auto, 1 SIMT, 2 Tensor-Core). GQA is handled inside the kernel
(no K/V repeat). The plugin picks the Tensor-Core kernel on sm_80+ when its
shared memory fits, otherwise the SIMT kernel; serialisation stores the three
attributes so engines are portable.

`build_engine.py` builds two engines for the same shape — ours and TensorRT's
builder-fused attention — checks ours against the fp32 PyTorch reference and
prints TFLOP/s for both plus `torch.sdpa`.
