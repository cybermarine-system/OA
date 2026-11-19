#!/usr/bin/env bash
# Qwen3-VL 공식 경로 엔진 빌드 (C++ llm_build + visual_build), Molmo2와 같은 프로파일 기준.
#   LLM: maxInputLen 2048, maxKVCacheCapacity 4096, maxBatchSize 1
#   시각부: minImageTokens 4(기본), maxImageTokens 1024, maxImageTokensPerImage 1024 (Molmo2 시각 엔진 상한 1024 pooled 토큰과 맞춤)
# 사용법: bash build_qwen3vl_engines.sh <작업 디렉터리 (onnx/ 포함)>   → <dir>/engines/{llm,visual}, <dir>/build_times.txt
set -euo pipefail
source /home/user/ondeviceAI/edgellm_check/scripts/env_cpp.sh
D=$1
s=$(date +%s)
$EDGELLM_BIN/llm/llm_build --onnxDir $D/onnx/llm --engineDir $D/engines/llm --maxInputLen 2048 --maxKVCacheCapacity 4096 --maxBatchSize 1 > $D/llm_build.log 2>&1
m=$(date +%s)
$EDGELLM_BIN/multimodal/visual_build --onnxDir $D/onnx/visual --engineDir $D/engines --maxImageTokens 1024 --maxImageTokensPerImage 1024 > $D/visual_build.log 2>&1
e=$(date +%s)
echo "llm_build_s=$((m-s)) visual_build_s=$((e-m))" | tee $D/build_times.txt
