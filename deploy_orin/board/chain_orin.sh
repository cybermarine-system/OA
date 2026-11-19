#!/bin/bash
# chain_orin.sh : Qwen4B 측정 종료 대기 -> 결과 복사 -> Qwen2B 측정 -> 결과 복사 -> GUI 복구 -> 전원 끔 (root 로 실행)
U=user; H=/home/$U
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
run_u(){ su - $U -c "export EDGELLM_BIN=$H/TensorRT-Edge-LLM-1231d64/build/examples EDGELLM_PLUGIN_PATH=$H/TensorRT-Edge-LLM-1231d64/build/libNvInfer_edgellm_plugin.so; $*"; }
copy(){ P=$1; B=$H/vrs_bundle_v2/$P
  run_u "ssh -o BatchMode=yes a100 mkdir -p /home/user/OA2/orin_results_v2/$P && cd $B && scp -o BatchMode=yes orin_results/*.json orin_results/*.log orin_results/*.txt build_llm.log build_visual.log a100:/home/user/OA2/orin_results_v2/$P/" && log "복사 완료 $P" || log "복사 실패 $P"; }

log "시작. Qwen4B 측정 종료 대기"
while pgrep -f "measure.sh qwen4b" >/dev/null; do sleep 60; done
log "Qwen4B 측정 끝"
copy molmo2_causal_r16_ep1_int4_v16k
copy qwen3vl4b_r16_ep1_int4_v16k
$H/bench/fix_clocks.sh
swapoff /swap16G 2>/dev/null; log "swap 상태: $(swapon --show | wc -l)개"
log "Qwen2B 측정 시작"
run_u "bash $H/v2/measure.sh qwen2b > $H/v2/measure_qwen2b.log 2>&1"
log "Qwen2B 측정 끝. 끝난 분할 수: $(grep -l 'Processing complete' $H/vrs_bundle_v2/qwen3vl2b_r16_ep1_int4_v16k/orin_results/*_c*.log 2>/dev/null | wc -l)/45"
copy qwen3vl2b_r16_ep1_int4_v16k
swapon /swap16G
systemctl set-default graphical.target
log "GUI 복구 설정 완료. 전원을 끈다"
sync; sleep 5; poweroff
