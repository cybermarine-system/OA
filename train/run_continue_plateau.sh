#!/usr/bin/env bash
# 직전 1 epoch 학습 결과(rank 16)에서 이어 학습. 학습률과 종료 시점은 검증 손실 개선 폭으로 결정함
#   500스텝마다 검증(840쌍). 0.5% 이상 개선이 2회 연속 없으면 학습률 절반, 1.25e-5에서도 없으면 종료. 상한은 추가 4 epoch(총 5 epoch)
#   causal 학습 → 평가 → 양방향 학습 → 평가 → 1 epoch 구간 체크포인트 평가(그래프용)
# 사용법: setsid nohup bash lora/run_continue_plateau.sh > lora/run_continue_plateau.log 2>&1 < /dev/null &
set -uo pipefail
ROOT=/home/user/ondeviceAI
PY=$ROOT/molmo2/.venv/bin/python
export HF_HOME=$ROOT/molmo2/hf_cache
cd $ROOT/molmo2
echo $$ > $ROOT/lora/run_continue_plateau.pid
step() { echo "=== [$(date '+%F %T')] $*"; }
train() { # $1=이름 $2.. 추가 옵션
  local name=$1; shift
  $PY $ROOT/lora/scripts/train_lora_vrsbench.py --out $ROOT/lora/runs/${name}_r16_plateau \
    --init_adapter $ROOT/lora/runs/${name}_r16/adapter --base_epoch 1 --shuffle_seed 1 \
    --sched plateau --lr 1e-4 --warmup 50 --plateau_factor 0.5 --plateau_patience 2 --plateau_threshold 0.005 \
    --min_lr 1.25e-5 --epochs 4 --eval_every 500 --bs 16 --grad_accum 2 --workers 16 "$@"
}
evaluate() { # $1=adapter $2=결과 이름 $3.. 추가 옵션
  local ad=$1 out=$2; shift 2
  $PY scripts/eval_vrsbench.py --n 2000 --bs 16 --prompt format_rules --lora $ad \
    --out results/vrsbench_2001_rules_molmo2_lora_${out}.jsonl "$@"
}
step "1/5 causal 이어 학습"
train causal
step "2/5 causal 평가 (epoch별, 검증 손실 최고 모델)"
for d in $ROOT/lora/runs/causal_r16_plateau/adapter_epoch* $ROOT/lora/runs/causal_r16_plateau/adapter_best; do
  [ -d "$d" ] && evaluate $d causal_r16_plateau_$(basename $d)_causal --causal
done
step "3/5 양방향 이어 학습"
train bidir --bidirectional
step "4/5 양방향 평가 (양방향 실행과 causal 실행)"
for d in $ROOT/lora/runs/bidir_r16_plateau/adapter_epoch* $ROOT/lora/runs/bidir_r16_plateau/adapter_best; do
  [ -d "$d" ] || continue
  evaluate $d bidir_r16_plateau_$(basename $d)_bidir
  evaluate $d bidir_r16_plateau_$(basename $d)_causal --causal
done
step "5/5 1 epoch 구간 체크포인트 평가 (그래프용)"
for s in 500 1000 1500 2000 2500; do
  evaluate $ROOT/lora/runs/causal_r16/adapter_step$s causal_r16_step${s}_causal --causal
  evaluate $ROOT/lora/runs/bidir_r16/adapter_step$s bidir_r16_step${s}_bidir
done
step "done"
rm -f $ROOT/lora/run_continue_plateau.pid
