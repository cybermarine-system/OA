#!/usr/bin/env bash
# A100 기준값: Qwen3-VL-2B FP16 + 어휘 축소(v16k), Orin 설정 언어 엔진(1024/1152/1) + 정합성 2,000 (분할 20개)
#   quant/bundle_check_v16k.sh 와 같은 방식. 시각 엔진은 그 사전 확인에서 같은 FP16 시각 ONNX를 Orin 설정(256토큰)으로 빌드한 것을 읽기만 함
set -uo pipefail
R=/home/user/ondeviceAI
ONNX=/home/user/OA2/out/qwen3vl2b_r16_ep1_fp16_v16k/llm
OUT=/home/user/OA2/orin_results_v2/ref_a100/qwen3vl2b_fp16_v16k
E=$OUT/engines
VIS=$R/quant/bundle_check_v16k/qwen3vl2b/engines/visual
step() { echo "=== [$(date '+%F %T')] $*"; }
die() { echo "!!! 실패: $*"; exit 1; }
mkdir -p $E
export TMPDIR=/home/user/OA2/tmp/export_tmp
source $R/edgellm_check/scripts/env_cpp.sh

nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 500 > $OUT/nvidia_smi_mem.csv & NS=$!
trap 'kill $NS 2>/dev/null' EXIT

step "언어 엔진 빌드 1024/1152/1"
[ -f $E/llm/config.json ] || /usr/bin/time -v $EDGELLM_BIN/llm/llm_build --onnxDir $ONNX --engineDir $E/llm \
  --maxInputLen 1024 --maxKVCacheCapacity 1152 --maxBatchSize 1 > $OUT/build_llm.log 2>&1 || die llm_build
grep -E "Total Weights Memory|LLM engine built successfully|Copied vocab_map" $OUT/build_llm.log
du -sb $E/llm $E/llm/llm.engine | tee $OUT/engine_bytes.txt

step "정합성 분할 20개 (deploy/orin/common 에서 읽기만, 출력은 OA2)"
cd $R/deploy/orin/common
for k in $(seq -w 1 20); do
  $EDGELLM_BIN/llm/llm_inference --engineDir $E/llm --multimodalEngineDir $VIS --inputFile chunks/cons_c$k.json \
    --outputFile $OUT/cons_c$k.json --dumpProfile --profileOutputFile $OUT/cons_c${k}_profile.json > $OUT/cons_c$k.log 2>&1 || die "infer cons_c$k"
done
python3 -I $R/deploy/orin/common/merge_outputs.py $OUT 20 $OUT/consistency2000_out.json
step done
