"""LoRA adapter를 Molmo2-4B HF 체크포인트에 병합해, 원본과 같은 파일 구조의 체크포인트로 저장.

- 원본 shard 이름·텐서 이름·index를 그대로 유지 (Qwen3 변환 스크립트와 HF 로더가 그대로 읽음)
- W' = W + (alpha / r) * B @ A, fp32로 계산 후 원래 dtype(BF16)으로 저장
- 코드·설정·토크나이저·전처리 파일은 복사

index 없는 단일 model.safetensors 체크포인트도 처리 (2026-11-09)

usage: python merge_lora_to_hf.py <molmo2_hf_dir> <adapter_dir> <out_dir>
"""
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

src, adp, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)

cfg = json.load(open(adp / "adapter_config.json"))
scale = cfg["lora_alpha"] / cfg["r"]
assert not cfg.get("use_dora") and not cfg.get("use_rslora"), "DoRA/rsLoRA 미지원"

A, B = {}, {}
with safe_open(adp / "adapter_model.safetensors", "pt") as f:
    for k in f.keys():
        base = k.replace("base_model.model.", "", 1)
        if ".lora_A." in base:
            A[base.replace(".lora_A.weight", ".weight")] = f.get_tensor(k)
        elif ".lora_B." in base:
            B[base.replace(".lora_B.weight", ".weight")] = f.get_tensor(k)
        else:
            raise SystemExit(f"예상 밖 adapter 텐서: {k}")
assert A.keys() == B.keys()
print(f"LoRA 대상 {len(A)}개, scale {scale}")

for p in src.iterdir():
    if not p.name.endswith(".safetensors"):
        shutil.copy2(p.resolve(), out / p.name)

if (src / "model.safetensors.index.json").exists():
    weight_map = json.load(open(src / "model.safetensors.index.json"))["weight_map"]
else:  # 단일 파일 체크포인트 (예: Qwen3-VL-2B) — 2026-11-09 추가
    weight_map = {"_": "model.safetensors"}
done = set()
for shard in sorted(set(weight_map.values())):
    tensors = {}
    with safe_open(src / shard, "pt") as f:
        meta = f.metadata() or {"format": "pt"}
        for k in f.keys():
            t = f.get_tensor(k)
            if k in A:
                delta = (B[k].float() @ A[k].float()) * scale
                assert delta.shape == t.shape, (k, delta.shape, t.shape)
                t = (t.float() + delta).to(t.dtype)
                done.add(k)
            tensors[k] = t.contiguous()
    save_file(tensors, out / shard, metadata=meta)
    print("saved", shard)
missing = set(A) - done
assert not missing, f"병합 안 된 텐서: {sorted(missing)[:5]}"
json.dump({"base": str(src), "adapter": str(adp), "scale": scale, "merged_tensors": len(done)},
          open(out / "merge_info.json", "w"), indent=1)
print("done", len(done))
