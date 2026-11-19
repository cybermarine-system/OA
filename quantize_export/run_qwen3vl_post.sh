#!/usr/bin/env bash
# 5·6단계: Qwen3-VL LoRA 학습 후 처리 (학습 결과 lora/runs/qwen3vl<SIZE>_r16/adapter 필요)
#   병합(lora/merged/qwen3vl<SIZE>_r16_ep1) → PyTorch 2,001 평가 → [FP16] 공식 export → 엔진 → 2,001
#   → [INT4] AWQ(VRSBench 보정 512, 텍스트 보정 강제) → export → 엔진 → 2,001 → FP16 대비 비교
# 사용법: setsid nohup bash quant/run_qwen3vl_post.sh 4b > quant/qwen3vl4b_r16_ep1/run.log 2>&1 < /dev/null &
#   단계 건너뛰기: 각 단계는 산출물이 있으면 건너뜀. SKIP_HF=1이면 PyTorch 평가 생략(나중에 따로)
set -uo pipefail
S=$1   # 4b | 2b
R=/home/user/ondeviceAI
PY=$R/molmo2/.venv/bin/python
case $S in
  4b) BASE=$R/edgellm_check/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17 ;;
  2b) BASE=$R/edgellm_check/hf_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203 ;;
esac
ADP=$R/lora/runs/qwen3vl${S}_r16/adapter
M=$R/lora/merged/qwen3vl${S}_r16_ep1
W=$R/quant/qwen3vl${S}_r16_ep1
RES=$R/molmo2/results
TAG=qwen3vl${S}_lora_r16_ep1
REQ=$R/edgellm_check/ws_qwen/vrs2001_requests.json
SMP=$R/edgellm_check/ws_qwen/vrs2001_requests_samples.jsonl
mkdir -p $W
step() { echo "=== [$(date '+%F %T')] $*"; }
die() { echo "!!! 실패: $*"; exit 1; }
edge() { ( source $R/.venv-edgellm/bin/activate; export LD_LIBRARY_PATH=$R/edgellm_libs HF_HOME=$R/edgellm_check/hf_cache; cd $W && "$@" ); }

step "1 LoRA 병합"
[ -f $M/merge_info.json ] || $PY $R/lora/scripts/merge_lora_to_hf.py $BASE $ADP $M || die merge

if [ "${SKIP_HF:-0}" != 1 ] && [ ! -f $RES/vrsbench_2001_rules_${TAG}_merged_hf.jsonl ]; then
  step "2 PyTorch 병합본 2,001 평가 (batch 1, greedy, 48토큰)"
  HF_HOME=$R/edgellm_check/hf_cache $PY $R/edgellm_check/scripts/eval_vrsbench_qwen3vl.py --model $M \
    --out $RES/vrsbench_2001_rules_${TAG}_merged_hf.jsonl > $W/hf_eval.log 2>&1 || die hf_eval
  tail -14 $W/hf_eval.log
fi

for P in fp16 int4; do
  D=$W/$P
  mkdir -p $D
  if [ $P = int4 ] && [ ! -f $D/int4_awq/calib_info.json ]; then
    step "int4: AWQ 양자화 (vrsbench_calib512_qwen3vl${S}, 텍스트 보정 강제)"
    edge python $R/quant/scripts/edgellm_quantize_vrs.py --force_text_calib -- llm --model_dir $M \
      --output_dir $D/int4_awq --quantization int4_awq --text_dataset vrsbench_calib512_qwen3vl${S} > $D/quantize.log 2>&1 || die quantize
    grep -E "calibration dataset|Quantization:|\[vrs\]" $D/quantize.log
  fi
  SRCM=$M; [ $P = int4 ] && SRCM=$D/int4_awq
  if [ ! -f $D/onnx/llm/model.onnx ]; then
    step "$P: export"
    edge tensorrt-edgellm-export $SRCM $D/onnx > $D/export.log 2>&1 || die export_$P
  fi
  if [ ! -f $D/build_times.txt ]; then
    step "$P: 엔진 빌드 (llm 2048/4096/1, visual 1024/1024)"
    bash $R/edgellm_check/scripts/build_qwen3vl_engines.sh $D || die build_$P
  fi
  if [ ! -f $RES/vrsbench_2001_rules_${TAG}_edgellm_$P.jsonl ]; then
    step "$P: Edge-LLM 2,001 추론"
    cd $R
    ( source $R/edgellm_check/scripts/env_cpp.sh
      $EDGELLM_BIN/llm/llm_inference --engineDir $D/engines/llm --multimodalEngineDir $D/engines/visual \
        --inputFile $REQ --outputFile $D/out2001.json --dumpProfile --profileOutputFile $D/profile2001.json ) > $D/infer2001.log 2>&1 || die infer_$P
    $PY $R/edgellm_check/scripts/edgellm_out_to_jsonl.py $SMP $D/out2001.json $RES/vrsbench_2001_rules_${TAG}_edgellm_$P.jsonl
  fi
done

step "비교"
H=$RES/vrsbench_2001_rules_${TAG}_merged_hf.jsonl
[ -f $H ] && $PY $R/quant/scripts/compare_pair.py $H $RES/vrsbench_2001_rules_${TAG}_edgellm_fp16.jsonl \
  --names PyTorch FP16 --json $W/compare_hf_fp16.json | tee $W/compare_hf_fp16.txt
$PY $R/quant/scripts/compare_pair.py $RES/vrsbench_2001_rules_${TAG}_edgellm_fp16.jsonl \
  $RES/vrsbench_2001_rules_${TAG}_edgellm_int4.jsonl --names FP16 INT4 --json $W/compare_fp16_int4.json | tee $W/compare_fp16_int4.txt
du -sb $M $W/fp16/onnx $W/fp16/engines/* $W/int4/int4_awq $W/int4/onnx $W/int4/engines/* > $W/sizes_bytes.txt
step done
