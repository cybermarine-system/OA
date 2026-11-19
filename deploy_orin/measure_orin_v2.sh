#!/usr/bin/env bash
# Orin Nano 측정 v2 (2026-11-16): 요청을 100개 단위로 나눠 프로세스마다 실행 (메모리 절감)
#   - 효율: 반복 5회 × 분할 5개(chunks/eff_c1..c5 = 워밍업 10 + 100문항), 최대 생성 16토큰. 각 분할 앞 10개는 지연에서 제외
#   - 정합성: 분할 20개(chunks/cons_c01..c20 = 100문항씩), 최대 생성 48토큰 → consistency2000_out.json으로 합침
#   - 배치 1, greedy(top_k 1). tegrastats 100ms 기록 → summarize_orin_v2.py
#   - v1(measure_orin.sh)은 510·2,000개를 한 프로세스에 넣어 CPU 메모리가 1.6~3.1 GB까지 늘어남 (A100 측정)
# 사용법 (패키지 폴더에서, 스왑 끈 상태):
#   bash ../common/measure_orin_v2.sh <LLM 엔진 폴더> <시각 엔진 폴더> <결과 폴더>
#   CONS_ONLY=1 이면 정합성만, EFF_ONLY=1 이면 효율만
set -euo pipefail
LLM=$1; VIS=$2; OUT=$3
C=$(cd "$(dirname "$0")" && pwd)
BIN=${EDGELLM_BIN:?EDGELLM_BIN(빌드한 examples 폴더, 예: ~/TensorRT-Edge-LLM/build/examples) 설정 필요}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
cd "$C"   # 요청 파일의 이미지 경로(images/...)가 common/ 기준

if [ "$(swapon --noheadings | wc -l)" != 0 ]; then echo "스왑이 켜져 있음. 측정 전 sudo swapoff -a"; exit 1; fi
{ nvpmodel -q 2>/dev/null; jetson_clocks --show 2>/dev/null | head -5; uname -a; free -m; } > "$OUT/env.txt" || true

run() { # $1=요청 파일 $2=출력 이름
  /usr/bin/time -v "$BIN/llm/llm_inference" --engineDir "$LLM" --multimodalEngineDir "$VIS" \
    --inputFile "$1" --outputFile "$OUT/$2.json" --dumpProfile --profileOutputFile "$OUT/$2_profile.json" \
    > "$OUT/$2.log" 2>&1 || { echo "!!! 실패: $2 (로그 $OUT/$2.log 마지막 줄)"; tail -5 "$OUT/$2.log"; exit 1; }
}

tegrastats --interval 100 --logfile "$OUT/tegrastats.log" & TS=$!
trap 'kill $TS 2>/dev/null || true' EXIT
echo "idle_start $(date +%s.%N)" >> "$OUT/marks.txt"; sleep 30; echo "idle_end $(date +%s.%N)" >> "$OUT/marks.txt"

if [ -z "${CONS_ONLY:-}" ]; then
  for i in 1 2 3 4 5; do
    echo "rep${i}_start $(date +%s.%N)" >> "$OUT/marks.txt"
    for k in 1 2 3 4 5; do run chunks/eff_c$k.json eff_rep${i}_c$k; done
    echo "rep${i}_end $(date +%s.%N)" >> "$OUT/marks.txt"
    echo "효율 반복 $i/5 끝"
    sleep 30
  done
fi

if [ -z "${EFF_ONLY:-}" ]; then
  echo "cons_start $(date +%s.%N)" >> "$OUT/marks.txt"
  for k in $(seq -w 1 20); do run chunks/cons_c$k.json cons_c$k; done
  echo "cons_end $(date +%s.%N)" >> "$OUT/marks.txt"
  python3 "$C/merge_outputs.py" "$OUT" 20 "$OUT/consistency2000_out.json"
fi
echo "완료: python3 $C/summarize_orin_v2.py $OUT"
