#!/usr/bin/env bash
# Orin 배포 패키지 v2(INT4 + 어휘 축소) 사전 확인 (A100, 2026-11-16)
#   패키지 폴더(심볼릭 링크)로 Orin과 같은 프로파일 엔진 빌드 → common/에서 분할 요청 실행 → 합치기 → A100 2048 엔진 결과와 비교
#   언어: 1,024 / 1,152 / 1 (세 모델 같음)
#   (Molmo2 시각 엔진은 런타임과 같은 TensorRT 11.4가 든 .venv-edgellm으로 빌드)
#   시각: Molmo2 crop 최대 5 / pooled 최대 725, Qwen maxImageTokens 256 (512x512 영상의 실제 크기)
# 사용법: bash quant/bundle_check_v16k.sh molmo2|qwen3vl4b|qwen3vl2b > quant/bundle_check_v16k_<m>.log 2>&1
set -uo pipefail
M=$1
R=/home/user/ondeviceAI
PY=$R/molmo2/.venv/bin/python
D=$R/deploy/orin
case $M in
  molmo2) B=$D/molmo2_causal_r16_ep1_int4_v16k; TAG=molmo2_merged_causal_r16_ep1 ;;
  *) B=$D/${M}_r16_ep1_int4_v16k; TAG=${M}_lora_r16_ep1 ;;
esac
OUT=$R/quant/bundle_check_v16k/$M
E=$OUT/engines
mkdir -p $E
step() { echo "=== [$(date '+%F %T')] $*"; }
die() { echo "!!! 실패: $*"; exit 1; }
source $R/edgellm_check/scripts/env_cpp.sh

step "언어 엔진 빌드 1024/1152/1"
[ -f $E/llm/config.json ] || $EDGELLM_BIN/llm/llm_build --onnxDir $B/llm --engineDir $E/llm \
  --maxInputLen 1024 --maxKVCacheCapacity 1152 --maxBatchSize 1 > $OUT/build_llm.log 2>&1 || die llm_build

step "시각 엔진 빌드"
if [ $M = molmo2 ]; then
  mkdir -p $E/visual; cp $B/visual/config.json $B/visual/new_embedding_fp16.safetensors $E/visual/
  [ -f $E/visual/visual.engine ] || $R/.venv-edgellm/bin/python $B/visual/build_check_visual_trt.py $B/visual/molmo2_visual_fp16.onnx $B/visual/ref_io.npz \
    $E/visual/visual.engine --max_crops 5 --max_pooled 725 > $OUT/build_visual.log 2>&1 || die visual_build
else
  [ -f $E/visual/config.json ] || $EDGELLM_BIN/multimodal/visual_build --onnxDir $B/visual --engineDir $E \
    --maxImageTokens 256 --maxImageTokensPerImage 256 > $OUT/build_visual.log 2>&1 || die visual_build
fi
tail -3 $OUT/build_visual.log
du -sb $E/llm $E/visual | tee $OUT/engine_bytes.txt

step "분할 실행: 효율 분할 1개 + 정합성 20개 (common/ 에서)"
cd $D/common
run() { $EDGELLM_BIN/llm/llm_inference --engineDir $E/llm --multimodalEngineDir $E/visual --inputFile $1 \
  --outputFile $OUT/$2.json --dumpProfile --profileOutputFile $OUT/$2_profile.json > $OUT/$2.log 2>&1 || die "infer $2"; }
run chunks/eff_c1.json eff_c1
for k in $(seq -w 1 20); do run chunks/cons_c$k.json cons_c$k; done
python3 merge_outputs.py $OUT 20 $OUT/consistency2000_out.json

step "A100 2048 엔진(어휘 축소) 결과와 비교"
cd $R
$PY edgellm_check/scripts/edgellm_out_to_jsonl.py deploy/orin/common/consistency2000_requests_samples.jsonl \
  $OUT/consistency2000_out.json $OUT/consistency2000_orinprofile.jsonl
$PY quant/scripts/compare_pair.py molmo2/results/vrsbench_2001_rules_${TAG}_edgellm_int4_v16k.jsonl $OUT/consistency2000_orinprofile.jsonl \
  --names A100_2048 OrinProfile --json $OUT/compare.json | head -3 | tee $OUT/compare.txt
python3 - <<EOF
import json,glob
ps=[json.load(open(p)) for p in sorted(glob.glob("$OUT/*_profile.json"))]
c=[p["peak_cpu_memory_mb"] for p in ps]; g=[p["peak_gpu_memory_mb"] for p in ps]
r=dict(n_runs=len(ps), cpu_peak_mb_max=max(c), cpu_peak_mb_min=min(c), gpu_peak_mb_max=max(g))
json.dump(r,open("$OUT/mem_summary.json","w"),indent=1); print(r)
EOF
step done
