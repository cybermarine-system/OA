#!/bin/bash
# Qwen3-VL-2B FP16 v16k: 풀기 → 링크 → 언어 엔진 빌드 → 판정
V2=~/vrs_bundle_v2; P=qwen3vl2b_r16_ep1_fp16_v16k; B=$V2/$P; I=$V2/qwen3vl2b_r16_ep1_int4_v16k
cd $V2 && tar -xf $V2/_recv_v3/qwen3vl2b_fp16_v16k.tar && ls $B/llm
[ -e $B/visual ] || ln -s $I/visual $B/visual
mkdir -p $B/engines; [ -e $B/engines/visual ] || ln -s $I/engines/visual $B/engines/visual
~/bench/fix_clocks.sh || exit 1
free -m | head -2
/usr/bin/time -v $EDGELLM_BIN/llm/llm_build --onnxDir $B/llm --engineDir $B/engines/llm \
  --maxInputLen 1024 --maxKVCacheCapacity 1152 --maxBatchSize 1 > $B/build_llm.log 2>&1
grep -E "built successfully|Copied vocab_map|Total Weights Memory|Error|Killed|insufficient" $B/build_llm.log | tail -8 | tee $B/engine_check.txt
grep -E "Elapsed|Maximum resident" $B/build_llm.log | tee -a $B/engine_check.txt
du -sh $B/engines/llm 2>/dev/null | tee -a $B/engine_check.txt
W=$(grep -oE "Total Weights Memory: [0-9]+" $B/build_llm.log | tail -1 | grep -oE "[0-9]+$")
if [ -n "$W" ] && [ "$W" -le 3200000000 ] && grep -q "built successfully" $B/build_llm.log; then
  echo "판정: 측정 진행 (가중치 ${W} B)" | tee $B/VERDICT.txt
else
  echo "판정: FP16 2B는 8GB 보드 탑재 불가 (가중치 ${W:-빌드 실패} B)" | tee $B/VERDICT.txt
fi
