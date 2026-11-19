#!/usr/bin/env bash
# run_ref_a100.sh 의 2단계(정합성 분할 20개)만. 엔진은 이미 빌드됨
set -uo pipefail
R=/home/user/ondeviceAI
OUT=/home/user/OA2/orin_results_v2/ref_a100/qwen3vl2b_fp16_v16k
E=$OUT/engines
VIS=$R/quant/bundle_check_v16k/qwen3vl2b/engines/visual
export TMPDIR=/home/user/OA2/tmp/export_tmp
source $R/edgellm_check/scripts/env_cpp.sh
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 500 > $OUT/nvidia_smi_mem_infer.csv & NS=$!
trap 'kill $NS 2>/dev/null' EXIT
cd $R/deploy/orin/common
echo "start $(date '+%F %T')"
for k in $(seq -w 1 20); do
  $EDGELLM_BIN/llm/llm_inference --engineDir $E/llm --multimodalEngineDir $VIS --inputFile chunks/cons_c$k.json \
    --outputFile $OUT/cons_c$k.json --dumpProfile --profileOutputFile $OUT/cons_c${k}_profile.json > $OUT/cons_c$k.log 2>&1 || { echo "!!! 실패 cons_c$k"; exit 1; }
  echo "cons_c$k 끝 $(date '+%T')"
done
PYTHONDONTWRITEBYTECODE=1 python3 -I $R/deploy/orin/common/merge_outputs.py $OUT 20 $OUT/consistency2000_out.json
echo "end $(date '+%F %T')"
