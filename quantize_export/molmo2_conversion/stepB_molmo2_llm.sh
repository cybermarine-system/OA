#!/usr/bin/env bash
# B단계: Molmo2-4B 언어 모델 부분(Qwen3 형식으로 변환)을 Edge-LLM에서 실행
#   FP16: 변환 체크포인트 -> ONNX export -> 엔진 빌드 -> 추론
#   INT4: 변환 체크포인트 -> INT4 AWQ 양자화 -> ONNX export -> 엔진 빌드 -> 추론
set -euo pipefail
ROOT=/home/user/ondeviceAI
WS=$ROOT/edgellm_check/ws/molmo2_llm
source $ROOT/.venv-edgellm/bin/activate
export LD_LIBRARY_PATH=$ROOT/edgellm_libs HF_HOME=$ROOT/edgellm_check/hf_cache
cd $WS

step() { echo "=== [$(date +%T)] $*"; }
build() {  # $1=checkpoint $2=engine dir
  tensorrt-edgellm-build --model-dir $1 --engine-dir $2 --components llm \
    --max-input-len 1024 --max-kv-cache-capacity 2048 --max-batch-size 1
}

step "fp16: export onnx";  tensorrt-edgellm-export $WS/hf $WS/onnx_fp16
step "fp16: build";        build $WS/hf $WS/engine_fp16
step "fp16: inference";    python $ROOT/edgellm_check/scripts/infer_text.py --engine-dir $WS/engine_fp16 \
                             --checkpoint-dir $WS/hf --out $WS/infer_fp16.json

step "int4: quantize";     tensorrt-edgellm-quantize llm --model_dir $WS/hf --output_dir $WS/int4_awq --quantization int4_awq
step "int4: export onnx";  tensorrt-edgellm-export $WS/int4_awq $WS/onnx_int4
step "int4: build";        build $WS/int4_awq $WS/engine_int4
step "int4: inference";    python $ROOT/edgellm_check/scripts/infer_text.py --engine-dir $WS/engine_int4 \
                             --checkpoint-dir $WS/int4_awq --out $WS/infer_int4.json
step "done"
du -sh $WS/hf $WS/int4_awq $WS/engine_fp16 $WS/engine_int4
