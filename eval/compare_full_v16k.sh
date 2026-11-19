#!/usr/bin/env bash
# test 전체 37,409문항: INT4 vs INT4+어휘 축소, FP16 vs INT4+어휘 축소 비교 (run_full_test.sh int4_v16k 끝난 뒤)
R=/home/user/ondeviceAI; PY=$R/molmo2/.venv/bin/python; RES=$R/molmo2/results
while kill -0 493268 2>/dev/null; do sleep 60; done
for m in molmo2 qwen3vl4b qwen3vl2b; do
  [ $m = molmo2 ] && T=molmo2_merged_causal_r16_ep1 || T=${m}_lora_r16_ep1
  echo "## $m"
  $PY $R/quant/scripts/compare_pair.py $RES/vrsbench_full_rules_${T}_edgellm_int4.jsonl $RES/vrsbench_full_rules_${T}_edgellm_int4_v16k.jsonl     --names INT4 INT4_v16k --json $R/quant/full/${m}_int4_vs_v16k.json
  $PY $R/quant/scripts/compare_pair.py $RES/vrsbench_full_rules_${T}_edgellm_fp16.jsonl $RES/vrsbench_full_rules_${T}_edgellm_int4_v16k.jsonl     --names FP16 INT4_v16k --json $R/quant/full/${m}_fp16_vs_v16k.json
done
echo "=== [$(date '+%F %T')] done"
