#!/usr/bin/env bash
# Qwen3-VL-2B LoRA 병합 FP16 + 어휘 축소(v16k) 언어 ONNX export (CPU 전용). INT4 v16k 번들(quant/run_qwen3vl_v16k.sh 1단계)과 같은 명령에서 체크포인트만 병합 FP16으로
set -euo pipefail
R=/home/user/ondeviceAI
O=/home/user/OA2/out/qwen3vl2b_r16_ep1_fp16_v16k
source $R/.venv-edgellm/bin/activate
export LD_LIBRARY_PATH=$R/edgellm_libs HF_HOME=/home/user/OA2/hf_cache TMPDIR=/home/user/OA2/tmp/export_tmp
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
cd $O
echo "start $(date '+%F %T')"
nice -n 10 ionice -c3 tensorrt-edgellm-export $R/lora/merged/qwen3vl2b_r16_ep1 $O --skip-visual --reduced-vocab-dir $R/quant/vocab/molmo2_v16k
echo "end $(date '+%F %T')"
