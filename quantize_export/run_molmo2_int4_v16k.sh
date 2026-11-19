#!/usr/bin/env bash
# 메모리 절감 시험 V2 (2026-11-16): 기존 Molmo2 ⑤ INT4 AWQ 체크포인트(lm_head FP16) + 출력 어휘 축소 16,384 (VRSBench train 기준)
#   export(--reduced-vocab-dir) → A100 엔진(2048/4096/1) → 2,001 → FP16·기존 INT4 대비 비교
set -uo pipefail
R=/home/user/ondeviceAI
PY=$R/molmo2/.venv/bin/python
SRC=$R/edgellm_check/ws_lora/causal_r16_ep1
Q=$R/quant/molmo2_causal_r16_ep1
W=$R/quant/molmo2_causal_r16_ep1_v16k
VOC=$R/quant/vocab/molmo2_v16k
V=$R/edgellm_check/ws_runner
RES=$R/molmo2/results
mkdir -p $W
step() { echo "=== [$(date '+%F %T')] $*"; }
die() { echo "!!! 실패: $*"; exit 1; }

step "1 ONNX export (어휘 축소)"
if [ ! -f $W/onnx_int4/llm/model.onnx ]; then
( source $R/.venv-edgellm/bin/activate; export LD_LIBRARY_PATH=$R/edgellm_libs HF_HOME=$R/edgellm_check/hf_cache
  cd $W && tensorrt-edgellm-export $Q/int4_awq $W/onnx_int4 --reduced-vocab-dir $VOC ) || die export
fi
$PY - <<PYEOF
import json; p="$W/onnx_int4/llm/config.json"; c=json.load(open(p)); c["image_token_id"]=151938; json.dump(c,open(p,"w"),indent=4)
PYEOF
[ -f $W/onnx_int4/llm/chat_template.jinja ] || cp $SRC/onnx_fp16/llm/chat_template.jinja $W/onnx_int4/llm/
ls -la $W/onnx_int4/llm/

step "2 엔진 빌드 (A100, 2048/4096/1)"
if [ ! -f $W/engine_int4/config.json ]; then
( source $R/edgellm_check/scripts/env_cpp.sh
  $EDGELLM_BIN/llm/llm_build --onnxDir $W/onnx_int4/llm --engineDir $W/engine_int4 \
    --maxInputLen 2048 --maxKVCacheCapacity 4096 --maxBatchSize 1 > $W/build_int4.log 2>&1 ) || die build
fi
ls -la $W/engine_int4/

step "3 Edge-LLM 2,001 추론"
cd $R
( source $R/edgellm_check/scripts/env_cpp.sh
  $EDGELLM_BIN/llm/llm_inference --engineDir $W/engine_int4 --multimodalEngineDir $V/molmo2_visual_engine_cpp \
    --inputFile $V/vrs2001/requests_cpp.json --outputFile $W/edgellm_out_int4.json --dumpProfile \
    --profileOutputFile $W/profile_int4.json ) > $W/infer_int4.log 2>&1 || die infer
$PY $R/edgellm_check/scripts/edgellm_out_to_jsonl.py $V/vrs2001/samples.jsonl $W/edgellm_out_int4.json \
  $RES/vrsbench_2001_rules_molmo2_merged_causal_r16_ep1_edgellm_int4_v16k.jsonl

step "4 비교"
$PY $R/quant/scripts/compare_pair.py $RES/vrsbench_2001_rules_molmo2_merged_causal_r16_ep1_edgellm_fp16.jsonl \
  $RES/vrsbench_2001_rules_molmo2_merged_causal_r16_ep1_edgellm_int4_v16k.jsonl --names FP16 INT4_v16k \
  --json $W/compare_fp16.json | tee $W/compare_fp16.txt
$PY $R/quant/scripts/compare_pair.py $RES/vrsbench_2001_rules_molmo2_merged_causal_r16_ep1_edgellm_int4.jsonl \
  $RES/vrsbench_2001_rules_molmo2_merged_causal_r16_ep1_edgellm_int4_v16k.jsonl --names INT4 INT4_v16k \
  --json $W/compare_int4.json | tee $W/compare_int4.txt
step done
