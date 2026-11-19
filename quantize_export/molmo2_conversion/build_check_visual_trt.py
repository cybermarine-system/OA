"""Molmo2 시각부 ONNX -> TensorRT 엔진 빌드 후, 참조 입력으로 PyTorch(FP32) 출력과 비교.

usage: python build_check_visual_trt.py <onnx> <ref_io.npz> <engine_out> [--fp32] [--max_crops 9] [--max_pooled 1024]
  (2026-11-16) --max_crops/--max_pooled: 프로파일 상한. VRSBench 512x512는 crop 5, pooled 725로 고정이라 5/725로 줄이면 작업 메모리 절감
"""
import sys
import time

import numpy as np
import tensorrt as trt
import torch

onnx_path, ref_path, engine_path = sys.argv[1:4]
fp16 = "--fp32" not in sys.argv
_arg = lambda k, d: int(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
MAXC, MAXP = _arg("--max_crops", 9), _arg("--max_pooled", 1024)
logger = trt.Logger(trt.Logger.WARNING)
ref = np.load(ref_path)
T, P = ref["images"].shape[1], ref["pooled_patches_idx"].shape[1]

builder = trt.Builder(logger)
network = builder.create_network(0 if int(trt.__version__.split(".")[0]) >= 10 else
                                 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
if not parser.parse_from_file(onnx_path):
    for i in range(parser.num_errors):
        print(parser.get_error(i))
    sys.exit(1)

config = builder.create_builder_config()
if fp16:
    # TensorRT 11은 strongly-typed 네트워크가 기본이라 FP16 플래그 대신 네트워크 dtype을 따름 -> 아래에서 처리
    if hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
profile = builder.create_optimization_profile()
# crop 수: T2(전역 1개)~T0(전역+분할, 최대 9개) / patch 토큰 수: 1~1024
profile.set_shape("images", (1, 1, 729, 588), (1, T, 729, 588), (1, MAXC, 729, 588))
profile.set_shape("pooled_patches_idx", (1, 1, 4), (1, P, 4), (1, MAXP, 4))
config.add_optimization_profile(profile)

t0 = time.time()
blob = builder.build_serialized_network(network, config)
assert blob is not None, "engine build failed"
open(engine_path, "wb").write(blob)
print(f"build {time.time() - t0:.0f}s, engine {blob.nbytes / 2**20:.0f} MiB, fp16_requested={fp16}, "
      f"fp16_flag_available={hasattr(trt.BuilderFlag, 'FP16')}")

engine = trt.Runtime(logger).deserialize_cuda_engine(blob)
ctx = engine.create_execution_context()
TORCH_DTYPE = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16, trt.DataType.INT64: torch.int64}
inputs = {"images": torch.from_numpy(ref["images"]).cuda(),
          "pooled_patches_idx": torch.from_numpy(ref["pooled_patches_idx"]).cuda()}
for name in list(inputs):
    inputs[name] = inputs[name].to(TORCH_DTYPE[engine.get_tensor_dtype(name)]).contiguous()
    ctx.set_input_shape(name, tuple(inputs[name].shape))
    ctx.set_tensor_address(name, inputs[name].data_ptr())
out_shape = tuple(ctx.get_tensor_shape("patch_embeds"))
out_dtype = TORCH_DTYPE[engine.get_tensor_dtype("patch_embeds")]
print("io dtypes:", {n: str(inputs[n].dtype) for n in inputs}, "out", out_dtype)
out = torch.empty(out_shape, dtype=out_dtype, device="cuda")
ctx.set_tensor_address("patch_embeds", out.data_ptr())
stream = torch.cuda.Stream()
for _ in range(3):
    ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()
t0 = time.time()
for _ in range(20):
    ctx.execute_async_v3(stream.cuda_stream)
stream.synchronize()
lat = (time.time() - t0) / 20 * 1000

out = out.float()
r = torch.from_numpy(ref["patch_embeds"]).cuda()
cos = torch.nn.functional.cosine_similarity(out, r, dim=-1)
print(f"out {out_shape}, max|diff| {(out - r).abs().max().item():.4f}, mean|diff| {(out - r).abs().mean().item():.5f}, "
      f"ref mean|x| {r.abs().mean().item():.4f}, cos min {cos.min().item():.5f} mean {cos.mean().item():.6f}, "
      f"latency(A100) {lat:.1f} ms")
