"""Molmo2-4B 시각부(SigLIP2 ViT 다층 특징 + 2x2 attention pooling + MLP projector)를 단일 ONNX로 export.

입력:  images [1, T, 729, 588] float32 (processor의 pixel_values, T=crop 수)
       pooled_patches_idx [1, P, 4] int64 (processor의 image_token_pooling, -1=무효)
출력:  patch_embeds [P, 2560] float32 = 영상 특징 + <im_patch> 토큰 임베딩
       (HF의 x[is_image_patch] += image_features를 미리 더한 값 → LLM은 해당 위치 임베딩을 교체만 하면 됨)

usage: python export_molmo2_visual.py <image> <out_dir> [--fp16]
  --fp16: 가중치·입출력을 FP16으로 export (TensorRT 10+ strongly-typed 빌드에서 FP16 엔진이 됨).
          참조 출력(ref_io.npz)은 항상 FP32로 계산.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

CKPT, REV = "allenai/Molmo2-4B", "042abfa7a38879a376cec03d949eff0aefaa0600"
img_path, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)


class Molmo2Visual(nn.Module):

    def __init__(self, vb, patch_token_embed):
        super().__init__()
        self.vb = vb
        self.register_buffer("patch_token_embed", patch_token_embed)

    def forward(self, images, pooled_patches_idx):
        vb = self.vb
        feats = vb.encode_image(images)
        dim = feats.shape[-1]
        valid = pooled_patches_idx >= 0
        to_pool = feats.reshape(-1, dim)[torch.clip(pooled_patches_idx[0], 0)]
        to_pool = to_pool * valid[0].to(to_pool.dtype)[:, :, None]
        attn_mask = valid.reshape([-1, 1, 1, valid.shape[-1]])
        denom = valid[0].to(to_pool.dtype).sum(-1).clamp(min=1)
        query = to_pool.sum(-2, keepdim=True) / denom[:, None, None]
        pooled = vb.image_pooling_2d(query, to_pool, attn_mask=attn_mask)
        pooled = vb.image_projector(pooled.reshape(pooled_patches_idx.shape[1], -1))
        return pooled + self.patch_token_embed


proc = AutoProcessor.from_pretrained(CKPT, revision=REV, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(CKPT, revision=REV, trust_remote_code=True,
                                                    dtype=torch.float32).eval()
cfg = model.config
img = Image.open(img_path).convert("RGB")
x = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": "Describe."},
                                                             {"type": "image", "image": img}]}],
                             tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
images = x["pixel_values"][None].float()
idx = x["image_token_pooling"][None].long()

wte = model.model.transformer.wte
patch_embed = wte.new_embedding[cfg.image_patch_id - wte.embedding.shape[0]].detach().float()
vis = Molmo2Visual(model.model.vision_backbone, patch_embed).eval()

with torch.inference_mode():
    ref_hf = model.model.vision_backbone(images, idx) + patch_embed
    ref = vis(images, idx)
print("wrapper vs HF vision_backbone max|diff|:", (ref - ref_hf).abs().max().item(), "shape", tuple(ref.shape))

np.savez(out / "ref_io.npz", images=images.numpy(), pooled_patches_idx=idx.numpy(), patch_embeds=ref.numpy())

fp16 = "--fp16" in sys.argv
if fp16:
    vis, images = vis.half(), images.half()
onnx_name = "molmo2_visual_fp16.onnx" if fp16 else "molmo2_visual.onnx"
torch.onnx.export(
    vis, (images, idx), str(out / onnx_name), dynamo=False, opset_version=17,
    input_names=["images", "pooled_patches_idx"], output_names=["patch_embeds"],
    dynamic_axes={"images": {1: "num_crops"}, "pooled_patches_idx": {1: "num_patches"},
                  "patch_embeds": {0: "num_patches"}},
)
print("exported", out / onnx_name)
