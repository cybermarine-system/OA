#!/bin/bash
# 사용법: bash ~/v2/build_visual.sh molmo2|qwen4b|qwen2b
case $1 in
  molmo2) P=molmo2_causal_r16_ep1_int4_v16k;;
  qwen4b) P=qwen3vl4b_r16_ep1_int4_v16k;;
  qwen2b) P=qwen3vl2b_r16_ep1_int4_v16k;;
  *) echo "인자: molmo2 | qwen4b | qwen2b"; exit 1;;
esac
B=~/vrs_bundle_v2/$P
~/bench/fix_clocks.sh || exit 1
mkdir -p $B/engines/visual
if [ $1 = molmo2 ]; then
  cp $B/visual/config.json $B/visual/new_embedding_fp16.safetensors $B/engines/visual/
  /usr/bin/time -v /usr/src/tensorrt/bin/trtexec --onnx=$B/visual/molmo2_visual_fp16.onnx \
    --saveEngine=$B/engines/visual/visual.engine --fp16 \
    --minShapes=images:1x1x729x588,pooled_patches_idx:1x1x4 \
    --optShapes=images:1x5x729x588,pooled_patches_idx:1x725x4 \
    --maxShapes=images:1x5x729x588,pooled_patches_idx:1x725x4 > $B/build_visual.log 2>&1
  grep -E "PASSED|insufficient|Error" $B/build_visual.log | head -5
else
  /usr/bin/time -v $EDGELLM_BIN/multimodal/visual_build --onnxDir $B/visual --engineDir $B/engines \
    --maxImageTokens 256 --maxImageTokensPerImage 256 > $B/build_visual.log 2>&1
  grep -E "Visual engine built successfully|Error" $B/build_visual.log | head -5
fi
grep -E "Elapsed|Maximum resident" $B/build_visual.log
ls -lh $B/engines/visual/
