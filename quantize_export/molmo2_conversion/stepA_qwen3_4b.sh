#!/usr/bin/env bash
# A단계: Edge-LLM 공식 지원 경로 기준선 (Qwen3-4B-Instruct-2507, INT4 AWQ)
#   HF 체크포인트 -> INT4 AWQ 양자화 -> ONNX export -> 엔진 빌드 -> 추론
set -euo pipefail
ROOT=/home/user/ondeviceAI
WS=$ROOT/edgellm_check/ws/qwen3_4b
source $ROOT/.venv-edgellm/bin/activate
export LD_LIBRARY_PATH=$ROOT/edgellm_libs HF_HOME=$ROOT/edgellm_check/hf_cache
mkdir -p $WS
cd $WS   # 소스 체크아웃 밖에서 실행해야 설치된 휠이 import됨

step() { echo "=== [$(date +%T)] $*"; }

step "download"
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir $WS/hf

step "quantize int4_awq"
tensorrt-edgellm-quantize llm --model_dir $WS/hf --output_dir $WS/int4_awq --quantization int4_awq

step "export onnx"
tensorrt-edgellm-export $WS/int4_awq $WS/onnx

step "build engine (direct builder)"
tensorrt-edgellm-build --model-dir $WS/int4_awq --engine-dir $WS/engine --components llm \
  --max-input-len 1024 --max-kv-cache-capacity 2048 --max-batch-size 1

step "inference"
python $ROOT/edgellm_check/scripts/infer_text.py --engine-dir $WS/engine --checkpoint-dir $WS/int4_awq \
  --out $WS/infer.json
step "done"
du -sh $WS/hf $WS/int4_awq $WS/onnx $WS/engine
