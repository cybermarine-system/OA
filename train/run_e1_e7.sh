#!/usr/bin/env bash
# E1 / E7 LoRA 학습·평가 파이프라인 (nohup으로 실행, 세션과 무관하게 진행)
#   1) causal LoRA 학습 (E7 (c), Edge-LLM 배포 조건) → VRSBench 2,001쌍 평가 (causal)
#   2) 양방향 LoRA 학습 (E1 / E7 (a), 원래 조건)      → 평가 (양방향, 그리고 causal로 실행했을 때)
# 사용법: setsid nohup bash /home/user/ondeviceAI/lora/run_e1_e7.sh > /home/user/ondeviceAI/lora/run_e1_e7.log 2>&1 < /dev/null &
set -uo pipefail
ROOT=/home/user/ondeviceAI
PY=$ROOT/molmo2/.venv/bin/python
export HF_HOME=$ROOT/molmo2/hf_cache
cd $ROOT/molmo2
echo $$ > $ROOT/lora/run_e1_e7.pid
step() { echo "=== [$(date '+%F %T')] $*"; }

step "1/5 causal LoRA 학습"
$PY $ROOT/lora/scripts/train_lora_vrsbench.py --out $ROOT/lora/runs/causal_r16 --bs 16 --grad_accum 2 --workers 16
step "2/5 causal LoRA 평가 (causal)"
$PY scripts/eval_vrsbench.py --n 2000 --bs 16 --prompt format_rules --causal \
  --lora $ROOT/lora/runs/causal_r16/adapter --out results/vrsbench_2001_rules_molmo2_lora_causal_r16_causal.jsonl

step "3/5 양방향 LoRA 학습"
$PY $ROOT/lora/scripts/train_lora_vrsbench.py --out $ROOT/lora/runs/bidir_r16 --bidirectional --bs 16 --grad_accum 2 --workers 16
step "4/5 양방향 LoRA 평가 (양방향)"
$PY scripts/eval_vrsbench.py --n 2000 --bs 16 --prompt format_rules \
  --lora $ROOT/lora/runs/bidir_r16/adapter --out results/vrsbench_2001_rules_molmo2_lora_bidir_r16_bidir.jsonl
step "5/5 양방향 LoRA를 causal로 실행 (학습·실행 조건 불일치 비교용)"
$PY scripts/eval_vrsbench.py --n 2000 --bs 16 --prompt format_rules --causal \
  --lora $ROOT/lora/runs/bidir_r16/adapter --out results/vrsbench_2001_rules_molmo2_lora_bidir_r16_causal.jsonl
step "done"
rm -f $ROOT/lora/run_e1_e7.pid
