#!/usr/bin/env bash
# A100 단독 지연 참고 측정: 6개 엔진(세 모델 × FP16·INT4)으로 2,001문항을 GPU 단독으로 순서대로 실행 (다른 GPU 작업 없을 때만)
# 결과: quant/latency_a100/<모델>_<정밀도>/{out.json,profile.json}. 요약은 quant/scripts/summarize_latency.py
# 사용법: setsid nohup bash quant/run_latency_a100.sh > quant/latency_a100/run.log 2>&1 < /dev/null &
set -uo pipefail
R=/home/user/ondeviceAI
cd $R
source $R/edgellm_check/scripts/env_cpp.sh
run() { # 이름 LLM VIS 요청
  local O=$R/quant/latency_a100/$1; mkdir -p $O
  echo "=== [$(date '+%F %T')] $1"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > $O/gpu_before.txt
  $EDGELLM_BIN/llm/llm_inference --engineDir $2 --multimodalEngineDir $3 --inputFile $4 --outputFile $O/out.json \
    --dumpProfile --profileOutputFile $O/profile.json > $O/infer.log 2>&1 || echo "!!! 실패 $1"
}
V=$R/edgellm_check/ws_runner
run molmo2_fp16 $R/edgellm_check/ws_lora/causal_r16_ep1/engine_fp16 $V/molmo2_visual_engine_cpp $V/vrs2001/requests_cpp.json
run molmo2_int4 $R/quant/molmo2_causal_r16_ep1/engine_int4 $V/molmo2_visual_engine_cpp $V/vrs2001/requests_cpp.json
for m in qwen3vl4b qwen3vl2b; do for p in fp16 int4; do
  run ${m}_$p $R/quant/${m}_r16_ep1/$p/engines/llm $R/quant/${m}_r16_ep1/$p/engines/visual $R/edgellm_check/ws_qwen/vrs2001_requests.json
done; done
echo "=== [$(date '+%F %T')] done"
