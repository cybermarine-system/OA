#!/usr/bin/env bash
# 7단계: test 전체 37,409문항 Edge-LLM 추론 (요청 quant/sets/test_full_requests.json, greedy, 최대 48토큰, batch 1)
# 사용법: setsid nohup bash quant/run_full_test.sh <모델> <정밀도> [<모델> <정밀도> ...] > quant/full/run_<이름>.log 2>&1 < /dev/null &
#   모델: molmo2 | qwen3vl4b | qwen3vl2b, 정밀도: fp16 | int4 | int4_v16k(어휘 축소). 여러 쌍을 주면 순서대로 실행
#   WAIT_PID=<pid> 를 주면 그 프로세스가 끝난 뒤 시작
set -uo pipefail
R=/home/user/ondeviceAI
PY=$R/molmo2/.venv/bin/python
REQ=$R/quant/sets/test_full_requests.json
SMP=$R/quant/sets/test_full_requests_samples.jsonl
[ -n "${WAIT_PID:-}" ] && while kill -0 $WAIT_PID 2>/dev/null; do sleep 30; done
cd $R   # 요청의 이미지 경로가 작업 폴더 루트 기준
while [ $# -ge 2 ]; do
  M=$1; P=$2; shift 2
  case $M in
    molmo2) TAG=molmo2_merged_causal_r16_ep1; VIS=$R/edgellm_check/ws_runner/molmo2_visual_engine_cpp
            [ $P = fp16 ] && LLM=$R/edgellm_check/ws_lora/causal_r16_ep1/engine_fp16 || LLM=$R/quant/molmo2_causal_r16_ep1/engine_int4 ;;
    qwen3vl4b|qwen3vl2b) TAG=${M}_lora_r16_ep1; LLM=$R/quant/${M}_r16_ep1/$P/engines/llm; VIS=$R/quant/${M}_r16_ep1/$P/engines/visual ;;
  esac
  # int4_v16k (2026-11-16): INT4 + 출력 어휘 16,384 축소. 시각 엔진은 INT4 때와 같은 FP16
  if [ $P = int4_v16k ]; then
    case $M in
      molmo2) LLM=$R/quant/molmo2_causal_r16_ep1_v16k/engine_int4 ;;
      *) LLM=$R/quant/${M}_r16_ep1/int4_v16k/engines/llm; VIS=$R/quant/${M}_r16_ep1/int4/engines/visual ;;
    esac
  fi
  O=$R/quant/full/${M}_$P
  OUT=$R/molmo2/results/vrsbench_full_rules_${TAG}_edgellm_$P.jsonl
  [ -f $OUT ] && { echo "skip $M $P"; continue; }
  mkdir -p $O
  echo "=== [$(date '+%F %T')] $M $P 시작 (LLM $LLM)"
  ( source $R/edgellm_check/scripts/env_cpp.sh
    $EDGELLM_BIN/llm/llm_inference --engineDir $LLM --multimodalEngineDir $VIS --inputFile $REQ \
      --outputFile $O/out.json --dumpProfile --profileOutputFile $O/profile.json ) > $O/infer.log 2>&1 \
    || { echo "!!! 실패 $M $P"; continue; }
  $PY $R/edgellm_check/scripts/edgellm_out_to_jsonl.py $SMP $O/out.json $OUT
  echo "=== [$(date '+%F %T')] $M $P 끝"
done
echo "=== [$(date '+%F %T')] done"
