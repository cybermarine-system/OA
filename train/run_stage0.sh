#!/usr/bin/env bash
# 경량화 0단계: 배포용 모델 ⑤(causal LoRA 1 epoch)의 FP16 기준점 확정
#   A. 병합 → 병합본 HF 평가(2,001) → Qwen3 형식 변환 → ONNX → Edge-LLM FP16 엔진(A100) → Edge-LLM 2,001 정합성
#   B. test 전체 37,409문항 HF 평가: ⑤ 병합본 → ② → ③ → ④ → ① → 3 epoch ⑤ ③ ④
# 사용법: setsid nohup bash lora/run_stage0.sh > lora/run_stage0.log 2>&1 < /dev/null &
set -uo pipefail
ROOT=/home/user/ondeviceAI
PY=$ROOT/molmo2/.venv/bin/python
export HF_HOME=$ROOT/molmo2/hf_cache
BASE=$ROOT/molmo2/hf_cache/hub/models--allenai--Molmo2-4B/snapshots/042abfa7a38879a376cec03d949eff0aefaa0600
M=$ROOT/lora/merged/molmo2_causal_r16_ep1
WS=$ROOT/edgellm_check/ws_lora/causal_r16_ep1
RES=$ROOT/molmo2/results
echo $$ > $ROOT/lora/run_stage0.pid
step() { echo "=== [$(date '+%F %T')] $*"; }
fail() { echo "!!! 실패: $*"; }
cd $ROOT/molmo2

step "A1 LoRA 병합"
[ -f $M/merge_info.json ] || $PY $ROOT/lora/scripts/merge_lora_to_hf.py $BASE $ROOT/lora/runs/causal_r16/adapter $M || fail A1

step "A2 병합본 HF 평가 (2,001, causal)"
$PY scripts/eval_vrsbench.py --n 2000 --bs 16 --prompt format_rules --causal --model_dir $M \
  --out $RES/vrsbench_2001_rules_molmo2_merged_causal_r16_ep1_causal.jsonl || fail A2
$PY scripts/compare_runs.py $RES/vrsbench_2001_rules_molmo2_lora_causal_r16_causal.jsonl \
  $RES/vrsbench_2001_rules_molmo2_merged_causal_r16_ep1_causal.jsonl | head -4

step "A3 Qwen3 형식 변환"
mkdir -p $WS
$PY $ROOT/edgellm_check/scripts/convert_molmo2_llm_to_qwen3.py $M $ROOT/edgellm_check/ws/qwen3_4b/hf/config.json $WS/hf || fail A3

step "A4 ONNX export (FP16)"
( source $ROOT/.venv-edgellm/bin/activate
  export LD_LIBRARY_PATH=$ROOT/edgellm_libs HF_HOME=$ROOT/edgellm_check/hf_cache
  cd $WS && tensorrt-edgellm-export $WS/hf $WS/onnx_fp16 ) || fail A4
# runner용: config.json에 image_token_id 추가 (기존 molmo2_onnx_llm과 같은 처리)
$PY - <<EOF
import json; p="$WS/onnx_fp16/llm/config.json"; c=json.load(open(p)); c["image_token_id"]=151938; json.dump(c,open(p,"w"),indent=4)
EOF
[ -f $WS/onnx_fp16/llm/chat_template.jinja ] || cp $ROOT/edgellm_check/ws/molmo2_llm/onnx_fp16/llm/chat_template.jinja $WS/onnx_fp16/llm/

step "A5 Edge-LLM FP16 엔진 빌드 (C++)"
( source $ROOT/edgellm_check/scripts/env_cpp.sh
  $EDGELLM_BIN/llm/llm_build --onnxDir $WS/onnx_fp16/llm --engineDir $WS/engine_fp16 \
    --maxInputLen 2048 --maxKVCacheCapacity 4096 --maxBatchSize 1 ) > $WS/build_fp16.log 2>&1 || fail A5
tail -3 $WS/build_fp16.log

step "A6 Edge-LLM 2,001 추론 (시각 엔진은 기존 FP16 그대로: LoRA는 언어 모델만 바꿈)"
V=$ROOT/edgellm_check/ws_runner
( source $ROOT/edgellm_check/scripts/env_cpp.sh
  $EDGELLM_BIN/llm/llm_inference --engineDir $WS/engine_fp16 --multimodalEngineDir $V/molmo2_visual_engine_cpp \
    --inputFile $V/vrs2001/requests_cpp.json --outputFile $WS/edgellm_out_fp16.json --dumpProfile ) > $WS/infer_fp16.log 2>&1 || fail A6
$PY $ROOT/edgellm_check/scripts/compare_edgellm_hf.py $V/vrs2001/samples.jsonl $WS/edgellm_out_fp16.json \
  --batch_ref $RES/vrsbench_2001_rules_molmo2_merged_causal_r16_ep1_causal.jsonl --out $WS/compare_fp16.jsonl || fail A6c

full() { # $1=결과 이름 $2.. 옵션
  local out=$1; shift
  [ -f $RES/vrsbench_full_rules_molmo2_$out.jsonl ] && { echo "skip $out"; return; }
  step "B test 전체: $out"
  $PY scripts/eval_vrsbench.py --n 0 --bs 16 --prompt format_rules --out $RES/vrsbench_full_rules_molmo2_$out.jsonl "$@" || fail "B $out"
}
L=$ROOT/lora/runs
full merged_causal_r16_ep1_causal --causal --model_dir $M          # ⑤ (배포 기준점)
full base_causal --causal                                           # ②
full lora_bidir_r16_bidir --lora $L/bidir_r16/adapter               # ③
full lora_bidir_r16_causal --causal --lora $L/bidir_r16/adapter     # ④
full base_bidir                                                     # ①
full lora_causal_r16_plateau_epoch3_causal --causal --lora $L/causal_r16_plateau/adapter_epoch3
full lora_bidir_r16_plateau_epoch3_bidir --lora $L/bidir_r16_plateau/adapter_epoch3
full lora_bidir_r16_plateau_epoch3_causal --causal --lora $L/bidir_r16_plateau/adapter_epoch3
step "done"
