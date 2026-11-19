#!/bin/bash
# 사용법: bash ~/demo/ask.sh <엔진> <이미지파일명> "<질문>" [free]
#   엔진: molmo2 | qwen4b | qwen2b | molmo2_full | qwen4b_full | qwen2b_full
#   free 를 붙이면 짧은 답 지시문 없이 자유 서술(최대 64토큰)
V2=~/vrs_bundle_v2
case $1 in
  molmo2) P=molmo2_causal_r16_ep1_int4_v16k;;  qwen4b) P=qwen3vl4b_r16_ep1_int4_v16k;;  qwen2b) P=qwen3vl2b_r16_ep1_int4_v16k;;
  molmo2_full) P=molmo2_causal_r16_ep1_int4_full;;  qwen4b_full) P=qwen3vl4b_r16_ep1_int4_full;;  qwen2b_full) P=qwen3vl2b_r16_ep1_int4_full;;
  *) echo "엔진: molmo2|qwen4b|qwen2b|molmo2_full|qwen4b_full|qwen2b_full"; exit 1;;
esac
B=$V2/$P; IMG=$2; Q=$3; MODE=${4:-short}
SUF=$'\nAnswer with as few words as possible. If the question asks how many, answer with a number only. If it is a yes/no question, answer yes or no only. Otherwise answer with a single word or short phrase.'
if [ "$MODE" = free ]; then SUF=""; MAXLEN=64; else MAXLEN=16; fi
TS=$(date +%H%M%S); REQ=~/demo/out/req_${1}_$TS.json; OUTJ=~/demo/out/out_${1}_$TS.json; LOG=~/demo/out/log_${1}_$TS.txt
python3 - "$IMG" "$Q$SUF" "$MAXLEN" "$REQ" << 'PY'
import json,sys
img,q,maxlen,req=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
d={"batch_size":1,"temperature":1.0,"top_p":1.0,"top_k":1,"max_generate_length":maxlen,"apply_chat_template":True,
   "requests":[{"messages":[{"role":"user","content":[{"type":"text","text":q},{"type":"image","image":"images/"+img}]}]}]}
json.dump(d,open(req,"w"),ensure_ascii=False,indent=1)
PY
cd $V2/common
echo "[$1] 엔진 적재 중..."
$EDGELLM_BIN/llm/llm_inference --engineDir $B/engines/llm --multimodalEngineDir $B/engines/visual --inputFile $REQ --outputFile $OUTJ > $LOG 2>&1 || { echo "실패. 로그:"; tail -5 $LOG; exit 1; }
python3 - "$OUTJ" "$1" "$IMG" "$Q" << 'PY'
import json,sys
o=json.load(open(sys.argv[1])); r=(o.get("responses") if isinstance(o,dict) else o)[0]
print(f"\n[{sys.argv[2]}] 이미지: {sys.argv[3]}\n질문: {sys.argv[4]}\n답변: {r['output_text']}\n시간: {r['request_wall_ms']:.0f} ms (생성 토큰 {r['generated_token_count']})\n")
PY
