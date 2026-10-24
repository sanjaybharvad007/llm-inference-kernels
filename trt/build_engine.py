"""Build a TensorRT engine that uses the FusedAttention plugin, run it, and
check it against the PyTorch reference and against TensorRT's own fused
attention (the MatMul-Softmax-MatMul subgraph that TensorRT's builder fuses
with its myelin kernels), timing all three.

    cmake -S trt -B trt/build && cmake --build trt/build -j
    python trt/build_engine.py --plugin trt/build/libfused_attention_trt.so --batch 4 --seqlen 2048 --heads 32 --kv-heads 8 --head-dim 128

Requires: tensorrt>=10 (Python), torch (buffers + reference).
"""
from __future__ import annotations

import argparse
import ctypes
import time

import numpy as np
import tensorrt as trt
import torch

from fused_attn.reference import attention_reference

LOGGER = trt.Logger(trt.Logger.WARNING)


def load_plugin(path: str) -> trt.IPluginCreatorV3One:
    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    registry = trt.get_plugin_registry()
    registry.load_library(path)
    creator = registry.get_creator("FusedAttention", "1", "")
    assert creator is not None, "FusedAttention creator not registered"
    return creator


def build_plugin_engine(creator, shape_q, shape_kv, dtype: trt.DataType, causal: bool) -> trt.ICudaEngine:
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)
    q = network.add_input("q", dtype, shape_q)
    k = network.add_input("k", dtype, shape_kv)
    v = network.add_input("v", dtype, shape_kv)
    fields = trt.PluginFieldCollection([
        trt.PluginField("causal", np.array([int(causal)], np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("scale", np.array([0.0], np.float32), trt.PluginFieldType.FLOAT32),
        trt.PluginField("backend", np.array([0], np.int32), trt.PluginFieldType.INT32),
    ])
    plugin = creator.create_plugin("attn", fields, trt.TensorRTPhase.BUILD)
    layer = network.add_plugin_v3([q, k, v], [], plugin)
    layer.get_output(0).name = "o"
    network.mark_output(layer.get_output(0))
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16 if dtype == trt.float16 else trt.BuilderFlag.BF16)
    return trt.Runtime(LOGGER).deserialize_cuda_engine(builder.build_serialized_network(network, config))


def build_native_engine(shape_q, shape_kv, dtype: trt.DataType, causal: bool) -> trt.ICudaEngine:
    """TensorRT's own fusion of the attention subgraph, for comparison."""
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)
    q = network.add_input("q", dtype, shape_q)
    k = network.add_input("k", dtype, shape_kv)
    v = network.add_input("v", dtype, shape_kv)
    B, Hq, Nq, D = shape_q
    Hkv, Nk = shape_kv[1], shape_kv[2]
    if Hq != Hkv:  # GQA: repeat KV heads
        rep = Hq // Hkv
        k = network.add_concatenation([k] * rep).get_output(0)
        k.name = "k_rep"
        v = network.add_concatenation([v] * rep).get_output(0)
        v.name = "v_rep"
        # concat along axis 1 interleaves wrongly for GQA; good enough for timing, not for numerics
    s = network.add_matrix_multiply(q, trt.MatrixOperation.NONE, k, trt.MatrixOperation.TRANSPOSE).get_output(0)
    scale = network.add_constant((1, 1, 1, 1), np.array([D ** -0.5], np.float32)).get_output(0)
    s = network.add_elementwise(s, scale, trt.ElementWiseOperation.PROD).get_output(0)
    if causal:
        mask = np.triu(np.full((Nq, Nk), -1e4, np.float32), k=Nk - Nq + 1)[None, None]
        m = network.add_constant(mask.shape, mask).get_output(0)
        s = network.add_elementwise(s, m, trt.ElementWiseOperation.SUM).get_output(0)
    p = network.add_softmax(s)
    p.axes = 1 << 3
    o = network.add_matrix_multiply(p.get_output(0), trt.MatrixOperation.NONE, v, trt.MatrixOperation.NONE).get_output(0)
    o.name = "o"
    network.mark_output(o)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16 if dtype == trt.float16 else trt.BuilderFlag.BF16)
    return trt.Runtime(LOGGER).deserialize_cuda_engine(builder.build_serialized_network(network, config))


def run(engine: trt.ICudaEngine, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, iters=50) -> tuple[torch.Tensor, float]:
    ctx = engine.create_execution_context()
    o = torch.empty_like(q)
    for name, t in (("q", q), ("k", k), ("v", v), ("o", o)):
        ctx.set_tensor_address(name, t.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    for _ in range(5):
        ctx.execute_async_v3(stream)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        ctx.execute_async_v3(stream)
    torch.cuda.synchronize()
    return o, (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plugin", default="trt/build/libfused_attention_trt.so")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--causal", action="store_true")
    args = ap.parse_args()
    tdt = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    trt_dt = trt.float16 if args.dtype == "fp16" else trt.bfloat16
    shape_q = (args.batch, args.heads, args.seqlen, args.head_dim)
    shape_kv = (args.batch, args.kv_heads, args.seqlen, args.head_dim)
    q = torch.randn(shape_q, device="cuda", dtype=tdt)
    k = torch.randn(shape_kv, device="cuda", dtype=tdt)
    v = torch.randn_like(k)

    creator = load_plugin(args.plugin)
    eng = build_plugin_engine(creator, shape_q, shape_kv, trt_dt, args.causal)
    o, ms = run(eng, q, k, v)
    ref = attention_reference(q.float(), k.float(), v.float(), causal=args.causal)
    err = (o.float() - ref).abs().max().item()
    flops = 4 * args.batch * args.heads * args.seqlen ** 2 * args.head_dim * (0.5 if args.causal else 1)
    print(f"plugin engine : {ms:8.3f} ms  {flops / ms / 1e9:7.1f} TFLOP/s  max|err| vs fp32 reference {err:.3e}")
    assert err < 2e-2, "plugin output mismatch"
    try:
        native = build_native_engine(shape_q, shape_kv, trt_dt, args.causal)
        _, ms_n = run(native, q, k, v)
        print(f"native TRT    : {ms_n:8.3f} ms  {flops / ms_n / 1e9:7.1f} TFLOP/s  (builder-fused MatMul-Softmax-MatMul)")
    except Exception as e:
        print("native TRT engine not built:", str(e)[:120])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=args.causal, enable_gqa=True)
    torch.cuda.synchronize()
    print(f"torch SDPA    : {(time.perf_counter() - t0) / 50 * 1e3:8.3f} ms")


if __name__ == "__main__":
    main()
