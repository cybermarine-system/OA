#!/bin/bash
# 어휘 축소 전(full vocab) 세 모델 측정 → A100 복사 → GUI 복구 → 전원 끔 (root로 실행)
U=user; H=/home/$U; V2=$H/vrs_bundle_v2
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
run_u(){ su - $U -c "export EDGELLM_BIN=$H/TensorRT-Edge-LLM-1231d64/build/examples EDGELLM_PLUGIN_PATH=$H/TensorRT-Edge-LLM-1231d64/build/libNvInfer_edgellm_plugin.so; $*"; }
copy(){ P=$1; B=$V2/$P
  run_u "ssh -o BatchMode=yes a100 mkdir -p /home/user/OA2/orin_results_v2/$P && cd $B && scp -o BatchMode=yes orin_results/*.json orin_results/*.log orin_results/*.txt build_llm.log a100:/home/user/OA2/orin_results_v2/$P/" && log "복사 완료 $P" || log "복사 실패 $P"; }
one(){ N=$1; P=$2; T=$3; B=$V2/$P
  $H/bench/fix_clocks.sh; swapoff /swap16G 2>/dev/null; log "swap: $(swapon --show | wc -l)개"
  log "$N 측정 시작 (제한 $T)"
  timeout -k 60 $T su - $U -c "export EDGELLM_BIN=$H/TensorRT-Edge-LLM-1231d64/build/examples EDGELLM_PLUGIN_PATH=$H/TensorRT-Edge-LLM-1231d64/build/libNvInfer_edgellm_plugin.so; bash $H/v2/measure.sh $N > $H/v2/measure_$N.log 2>&1"
  RC=$?; pkill -f llm_inference; pkill -x tegrastats
  D=$(grep -l 'Processing complete' $B/orin_results/*_c*.log 2>/dev/null | wc -l)
  log "$N 측정 끝 (rc=$RC). 끝난 분할 수: $D/45"
  if [ "$D" -lt 45 ]; then run_u "bash $V2/common/collect_fail_info.sh $B/orin_results"; log "$N 실패 정보 수집"; fi
  copy $P; }
log "시작"
one molmo2_full molmo2_causal_r16_ep1_int4_full 4h
one qwen4b_full qwen3vl4b_r16_ep1_int4_full 2h
one qwen2b_full qwen3vl2b_r16_ep1_int4_full 90m
swapon /swap16G; systemctl set-default graphical.target
log "GUI 복구 설정 완료. 전원을 끈다"; sync; sleep 5; poweroff
