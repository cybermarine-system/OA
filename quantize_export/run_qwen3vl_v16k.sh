#!/usr/bin/env bash
# 메모리 절감 V2를 Qwen3-VL에 같은 조건으로 적용 (2026-11-16)
#   기존 INT4 AWQ 체크포인트(quant/qwen3vl<S>_r16_ep1/int4/int4_awq) + Molmo2와 같은 vocab_map(quant/vocab/molmo2_v16k, 토큰 id 동일 확인)
#   export(--reduced-vocab-dir, 언어 모델만) → llm_build(2048/4096/1) → 시각 엔진은 기존 INT4 묶음의 FP16 시각 엔진 그대로 → 2,001 → 비교
# 사용법: setsid nohup bash quant/run_qwen3vl_v16k.sh 4b > quant/qwen3vl4b_r16_ep1/int4_v16k.run.log 2>&1 < /dev/null &
set -uo pipefail
S=$1
R=/home/user/ondeviceAI
PY=$R/molmo2/.venv/bin/python
W=$R/quant/qwen3vl${S}_r16_ep1
D=$W/int4_v16k
VOC=$R/quant/vocab/molmo2_v16k
RES=$R/molmo2/results
TAG=qwen3vl${S}_lora_r16_ep1
REQ=$R/edgellm_check/ws_qwen/vrs2001_requests.json
SMP=$R/edgellm_check/ws_qwen/vrs2001_requests_samples.jsonl
mkdir -p $D
step() { echo "=== [$(date '+%F %T')] $*"; }
die() { echo "!!! 실패: $*"; exit 1; }

step "1 export (언어 모델, 어휘 축소)"
if [ ! -f $D/onnx/llm/model.onnx ]; then
( source $R/.venv-edgellm/bin/activate; export LD_LIBRARY_PATH=$R/edgellm_libs HF_HOME=$R/edgellm_check/hf_cache
  cd $D && tensorrt-edgellm-export $W/int4/int4_awq $D/onnx --skip-visual --reduced-vocab-dir $VOC ) > $D/export.log 2>&1 || die export
fi
ls -la $D/onnx/llm/

step "2 llm_build (2048/4096/1)"
if [ ! -f $D/engines/llm/config.json ]; then
( source $R/edgellm_check/scripts/env_cpp.sh
  $EDGELLM_BIN/llm/llm_build --onnxDir $D/onnx/llm --engineDir $D/engines/llm --maxInputLen 2048 --maxKVCacheCapacity 4096 \
    --maxBatchSize 1 ) > $D/llm_build.log 2>&1 || die build
fi
ls -la $D/engines/llm/

step "3 Edge-LLM 2,001 추론 (시각 엔진: $W/int4/engines/visual)"
cd $R
( source $R/edgellm_check/scripts/env_cpp.sh
  $EDGELLM_BIN/llm/llm_inference --engineDir $D/engines/llm --multimodalEngineDir $W/int4/engines/visual \
    --inputFile $REQ --outputFile $D/out2001.json --dumpProfile --profileOutputFile $D/profile2001.json ) > $D/infer2001.log 2>&1 || die infer
$PY $R/edgellm_check/scripts/edgellm_out_to_jsonl.py $SMP $D/out2001.json $RES/vrsbench_2001_rules_${TAG}_edgellm_int4_v16k.jsonl

step "4 비교"
$PY $R/quant/scripts/compare_pair.py $RES/vrsbench_2001_rules_${TAG}_edgellm_fp16.jsonl $RES/vrsbench_2001_rules_${TAG}_edgellm_int4_v16k.jsonl \
  --names FP16 INT4_v16k --json $D/compare_fp16.json | tee $D/compare_fp16.txt
$PY $R/quant/scripts/compare_pair.py $RES/vrsbench_2001_rules_${TAG}_edgellm_int4.jsonl $RES/vrsbench_2001_rules_${TAG}_edgellm_int4_v16k.jsonl \
  --names INT4 INT4_v16k --json $D/compare_int4.json | tee $D/compare_int4.txt
du -sb $W/int4/engines/llm $D/engines/llm > $D/sizes_bytes.txt
step done
