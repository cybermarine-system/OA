#!/usr/bin/env bash
# GPU 작업 연쇄 실행기 (무거운 작업 동시 2개 이하: 줄 A·B 각각 순차)
#   A: <WAIT_PID 종료> → Qwen3-VL-2B LoRA 학습 → 2B 후처리(병합·평가·FP16/INT4 엔진) → 2B test 전체(FP16·INT4)
#   B: <WAIT_PID 종료> → Qwen3-VL-4B 후처리 → 4B test 전체(FP16·INT4)
# 사용법: WAIT_PID=<pid> setsid nohup bash quant/run_chain.sh A|B > quant/chain_<A|B>.log 2>&1 < /dev/null &
set -uo pipefail
R=/home/user/ondeviceAI
cd $R
[ -n "${WAIT_PID:-}" ] && while kill -0 $WAIT_PID 2>/dev/null; do sleep 30; done
step() { echo "=== [$(date '+%F %T')] $*"; }
case $1 in
  A)
    B2=$R/edgellm_check/hf_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203
    if [ ! -d $R/lora/runs/qwen3vl2b_r16/adapter ]; then
      step "2B 학습 시작"
      HF_HOME=$R/edgellm_check/hf_cache $R/molmo2/.venv/bin/python $R/lora/scripts/train_lora_qwen3vl.py --model $B2 \
        --out $R/lora/runs/qwen3vl2b_r16 > $R/lora/runs_qwen3vl2b_r16.log 2>&1 &
      echo $! > $R/lora/runs_qwen3vl2b_r16.pid; wait $!
      step "2B 학습 끝 (exit $?)"
    fi
    [ -d $R/lora/runs/qwen3vl2b_r16/adapter ] || { echo "!!! 2B adapter 없음"; exit 1; }
    mkdir -p $R/quant/qwen3vl2b_r16_ep1
    step "2B 후처리"; bash $R/quant/run_qwen3vl_post.sh 2b > $R/quant/qwen3vl2b_r16_ep1/run.log 2>&1
    step "2B test 전체"; bash $R/quant/run_full_test.sh qwen3vl2b fp16 qwen3vl2b int4 > $R/quant/full/run_qwen3vl2b.log 2>&1
    ;;
  B)
    [ -d $R/lora/runs/qwen3vl4b_r16/adapter ] || { echo "!!! 4B adapter 없음"; exit 1; }
    mkdir -p $R/quant/qwen3vl4b_r16_ep1
    step "4B 후처리"; bash $R/quant/run_qwen3vl_post.sh 4b > $R/quant/qwen3vl4b_r16_ep1/run.log 2>&1
    step "4B test 전체"; bash $R/quant/run_full_test.sh qwen3vl4b fp16 qwen3vl4b int4 > $R/quant/full/run_qwen3vl4b.log 2>&1
    ;;
esac
step done
