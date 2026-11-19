#!/bin/bash
# 사용법: 먼저 sudo swapoff /swap16G 를 직접 친 뒤
#        nohup bash ~/v2/measure.sh molmo2 > ~/v2/measure_molmo2.log 2>&1 &
case $1 in
  molmo2) P=molmo2_causal_r16_ep1_int4_v16k;;
  qwen4b) P=qwen3vl4b_r16_ep1_int4_v16k;;
  qwen2b) P=qwen3vl2b_r16_ep1_int4_v16k;;
  molmo2_full) P=molmo2_causal_r16_ep1_int4_full;;
  qwen4b_full) P=qwen3vl4b_r16_ep1_int4_full;;
  qwen2b_full) P=qwen3vl2b_r16_ep1_int4_full;;
  fp16_2b) P=qwen3vl2b_r16_ep1_fp16_v16k;;
  *) echo "인자: molmo2 | qwen4b | qwen2b"; exit 1;;
esac
B=~/vrs_bundle_v2/$P
if [ -n "$(swapon --show)" ]; then echo "스왑이 켜져 있음. sudo swapoff /swap16G 먼저"; exit 1; fi
free -m
cd $B && bash ../common/measure_orin_v2.sh $B/engines/llm $B/engines/visual $B/orin_results
echo "=== 실패한 분할 (출력 없으면 전부 성공) ==="
grep -LE "Processing complete: (110/110|100/100)" $B/orin_results/*_c*.log
python3 ../common/summarize_orin_v2.py $B/orin_results
echo "=== summary.json ==="; cat $B/orin_results/summary.json
echo "=== 끝. 이제 sudo swapon /swap16G 를 직접 칠 것 ==="
