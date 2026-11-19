#!/usr/bin/env bash
# 2단계: Molmo2 ⑤(causal LoRA rank 16, 1 epoch) 병합본 언어 모델 INT4 AWQ
#   양자화(VRSBench 보정 512) → ONNX → A100 엔진(2048/4096/1, FP16 엔진과 같은 프로파일) → 2,001문항 → FP16 엔진과 비교
# 사용법: setsid nohup bash quant/run_molmo2_int4.sh > quant/molmo2_causal_r16_ep1/run.log 2>&1 < /dev/null &
set -uo pipefail
R=/home/user/ondeviceAI
PY=$R/molmo2/.venv/bin/python
SRC=$R/edgellm_check/ws_lora/causal_r16_ep1
W=$R/quant/molmo2_causal_r16_ep1
V=$R/edgellm_check/ws_runner
mkdir -p $W
step() { echo "=== [$(date '+%F %T')] $*"; }
die() { echo "!!! 실패: $*"; exit 1; }

step "1 INT4 AWQ 양자화 (보정: vrsbench_calib512_molmo2)"
if [ ! -f $W/int4_awq/calib_info.json ]; then
( source $R/.venv-edgellm/bin/activate; export LD_LIBRARY_PATH=$R/edgellm_libs HF_HOME=$R/edgellm_check/hf_cache
  cd $W && python $R/quant/scripts/edgellm_quantize_vrs.py -- llm --model_dir $SRC/hf --output_dir $W/int4_awq \
    --quantization int4_awq --text_dataset vrsbench_calib512_molmo2 ) || die quantize
fi

step "2 ONNX export"
if [ ! -f $W/onnx_int4/llm/model.onnx ]; then
( source $R/.venv-edgellm/bin/activate; export LD_LIBRARY_PATH=$R/edgellm_libs HF_HOME=$R/edgellm_check/hf_cache
  cd $W && tensorrt-edgellm-export $W/int4_awq $W/onnx_int4 ) || die export
fi
$PY - <<EOF
import json; p="$W/onnx_int4/llm/config.json"; c=json.load(open(p)); c["image_token_id"]=151938; json.dump(c,open(p,"w"),indent=4)
EOF
[ -f $W/onnx_int4/llm/chat_template.jinja ] || cp $SRC/onnx_fp16/llm/chat_template.jinja $W/onnx_int4/llm/

step "3 엔진 빌드 (A100, maxInputLen 2048, KV 4096, batch 1)"
if [ ! -f $W/engine_int4/config.json ]; then
( source $R/edgellm_check/scripts/env_cpp.sh; s=$(date +%s)
  $EDGELLM_BIN/llm/llm_build --onnxDir $W/onnx_int4/llm --engineDir $W/engine_int4 \
    --maxInputLen 2048 --maxKVCacheCapacity 4096 --maxBatchSize 1 > $W/build_int4.log 2>&1; r=$?
  echo "llm_build_s=$(( $(date +%s) - s ))" > $W/build_times.txt; exit $r ) || die build
fi
tail -3 $W/build_int4.log

step "4 Edge-LLM 2,001 추론 (시각 엔진 FP16 기존 것, 요청 vrs2001/requests_cpp.json)"
cd $R
( source $R/edgellm_check/scripts/env_cpp.sh
  $EDGELLM_BIN/llm/llm_inference --engineDir $W/engine_int4 --multimodalEngineDir $V/molmo2_visual_engine_cpp \
    --inputFile $V/vrs2001/requests_cpp.json --outputFile $W/edgellm_out_int4.json --dumpProfile \
    --profileOutputFile $W/profile_int4.json ) > $W/infer_int4.log 2>&1 || die infer
$PY $R/edgellm_check/scripts/edgellm_out_to_jsonl.py $V/vrs2001/samples.jsonl $W/edgellm_out_int4.json \
  $R/molmo2/results/vrsbench_2001_rules_molmo2_merged_causal_r16_ep1_edgellm_int4.jsonl
[ -f $R/molmo2/results/vrsbench_2001_rules_molmo2_merged_causal_r16_ep1_edgellm_fp16.jsonl ] || \
$PY $R/edgellm_check/scripts/edgellm_out_to_jsonl.py $V/vrs2001/samples.jsonl $SRC/edgellm_out_fp16.json \
  $R/molmo2/results/vrsbench_2001_rules_molmo2_merged_causal_r16_ep1_edgellm_fp16.jsonl

step "5 FP16 엔진 대비 비교"
$PY $R/quant/scripts/compare_pair.py \
  $R/molmo2/results/vrsbench_2001_rules_molmo2_merged_causal_r16_ep1_edgellm_fp16.jsonl \
  $R/molmo2/results/vrsbench_2001_rules_molmo2_merged_causal_r16_ep1_edgellm_int4.jsonl \
  --names FP16 INT4 --json $W/compare_fp16_int4.json | tee $W/compare_fp16_int4.txt
du -sb $W/int4_awq $W/onnx_int4 $W/engine_int4 > $W/sizes_bytes.txt
step done
