#!/usr/bin/env bash
# setup_vrs_bundle_v2.sh : ~/vrs_bundle_v2(V2)를 1차 폴더 ~/vrs_bundle(V1)과 연결 (2026-11-16, OA2/transfer_v2)
#   - V2/common/ 에 V1/common 의 images/, 요청 JSON(*.json, *.jsonl), edgellm_patches/ 를 심볼릭 링크 (V2에 같은 이름이 있으면 V2 우선, 건너뜀)
#   - V2/<모델>_int4_v16k/visual -> V1/<모델>_int4/visual
#   - measure_orin_v2.sh 가 쓰는 ../common, images/ 상대 경로를 실제로 ls 로 검증. 빠진 것이 있으면 이름을 찍고 종료 코드 1
#   - 여러 번 실행해도 안전 (있는 링크는 건너뜀)
# 사용법: bash ~/vrs_bundle_v2/common/setup_vrs_bundle_v2.sh      (V1=... V2=... 로 경로 바꿀 수 있음)
set -u
V1=${V1:-$HOME/vrs_bundle}
V2=${V2:-$HOME/vrs_bundle_v2}
MODELS="molmo2_causal_r16_ep1 qwen3vl4b_r16_ep1 qwen3vl2b_r16_ep1"
MISSING=0
miss() { echo "  [없음] $1"; MISSING=$((MISSING + 1)); }

link() { # $1=대상(절대경로) $2=링크 경로
  local tgt=$1 lnk=$2
  if [ -L "$lnk" ]; then
    if [ "$(readlink "$lnk")" = "$tgt" ]; then echo "  [건너뜀] 이미 링크: $lnk"; else echo "  [건너뜀] 다른 곳을 가리키는 링크가 있음: $lnk -> $(readlink "$lnk")"; fi
  elif [ -e "$lnk" ]; then
    echo "  [건너뜀] V2 파일 우선: $lnk"
  elif [ ! -e "$tgt" ]; then
    miss "링크 대상 $tgt"
  else
    ln -s "$tgt" "$lnk" && echo "  [링크] $lnk -> $tgt"
  fi
}

echo "V1=$V1"; echo "V2=$V2"
[ -d "$V1/common" ] || { echo "V1/common 없음: $V1/common"; exit 1; }
[ -d "$V2/common" ] || { echo "V2/common 없음 (common_v2.tar를 V2에 풀었는지 확인): $V2/common"; exit 1; }
V1=$(cd "$V1" && pwd); V2=$(cd "$V2" && pwd)

echo "== 1. common 연결"
link "$V1/common/images" "$V2/common/images"
link "$V1/common/edgellm_patches" "$V2/common/edgellm_patches"
for f in "$V1"/common/*.json "$V1"/common/*.jsonl; do
  [ -e "$f" ] || continue
  link "$f" "$V2/common/$(basename "$f")"
done

echo "== 2. 모델별 visual 연결"
for m in $MODELS; do
  d="$V2/${m}_int4_v16k"
  if [ ! -d "$d" ]; then miss "모델 폴더 $d (해당 tar를 풀었는지 확인)"; continue; fi
  link "$V1/${m}_int4/visual" "$d/visual"
done

echo "== 3. 검증 (패키지 폴더에서 ../common, common 에서 images/)"
for m in $MODELS; do
  d="$V2/${m}_int4_v16k"; [ -d "$d" ] || continue
  echo "-- $m"
  before=$MISSING
  cd "$d" || exit 1
  if [ "${m#molmo2}" != "$m" ]; then vf="config.json molmo2_visual_fp16.onnx new_embedding_fp16.safetensors"
  else vf="config.json model.onnx model.onnx.data preprocessor_config.json"; fi
  for f in ../common/measure_orin_v2.sh ../common/summarize_orin_v2.py ../common/merge_outputs.py \
           ../common/chunks/eff_c{1,2,3,4,5}.json $(printf '../common/chunks/cons_c%02d.json ' $(seq 1 20)) \
           ../common/consistency2000_requests_samples.jsonl \
           llm/config.json llm/model.onnx llm/model.onnx.data llm/embedding.safetensors llm/vocab_map.safetensors \
           llm/tokenizer.json llm/tokenizer_config.json $(for x in $vf; do echo "visual/$x"; done); do
    ls "$f" > /dev/null 2>&1 || miss "$d/$f"
  done
  cd - > /dev/null
  [ "$MISSING" -eq "$before" ] && echo "  OK (../common, llm/, visual/ 모두 해석됨)"
done

echo "-- 요청 파일 안의 이미지 경로 (common 기준)"
cnt=0; bad=0
while read -r p; do
  cnt=$((cnt + 1))
  ls "$V2/common/$p" > /dev/null 2>&1 || { echo "  [없음] 이미지 $p"; bad=$((bad + 1)); }
done < <(grep -oh '"images/[^"]*"' "$V2"/common/chunks/*.json | tr -d '"' | sort -u)
echo "  이미지 $cnt개 확인, 없음 $bad개"
MISSING=$((MISSING + bad))
[ "$cnt" -gt 0 ] || { echo "  [없음] chunks 안 이미지 경로를 하나도 찾지 못함"; MISSING=$((MISSING + 1)); }

if [ "$MISSING" -gt 0 ]; then echo "실패: 빠진 항목 $MISSING개"; exit 1; fi
echo "완료: 연결·검증 통과"
