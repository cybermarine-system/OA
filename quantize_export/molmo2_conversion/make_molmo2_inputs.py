"""Molmo2 Edge-LLM runner용 입력 파일 생성 (+ 선택: 같은 입력의 HF causal 참조 답변).

HF processor로 만든 input_ids / pixel_values / image_token_pooling을 요청별 바이너리(<qid>.molmo2.bin)로 쓰고,
Edge-LLM llm_inference 입력 JSON(requests.json)을 만든다. 각 요청의 image 경로는 <out>/pre/<qid>.png
(원본 이미지 심볼릭 링크)이고, C++ Molmo2 runner는 확장자를 .molmo2.bin으로 바꿔 사이드카를 읽는다.

바이너리 형식 (little-endian): b"MOLMO2PP", uint32 version=1, uint32 n_arrays=3, 배열마다
  uint32 dtype(0=int32, 1=fp16, 2=int64), uint32 ndim, int64 dims[ndim], 원시 데이터
  순서: input_ids int32 [L], pixel_values fp16 [T,729,588], image_token_pooling int64 [P,4]

usage (molmo2/.venv, HF_HOME=molmo2/hf_cache):
  python make_molmo2_inputs.py --n 200 --out <dir> [--hf_ref] [--max_new_tokens 48]
  --n: eval_vrsbench.py와 같은 층화 추출(seed 0, n=2000) 결과의 앞쪽 n개를 사용
"""
import argparse
import json
import os
import struct
import sys
import time
from os.path import join

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, "/home/user/ondeviceAI/molmo2/scripts")
from eval_vrsbench import CKPT, PROMPTS, REVISION, ROOT, TYPE_MAP, score, stratified_sample  # noqa: E402

DTYPES = {np.dtype(np.int32): 0, np.dtype(np.float16): 1, np.dtype(np.int64): 2}


def write_sidecar(path, arrays):
    with open(path, "wb") as f:
        f.write(b"MOLMO2PP")
        f.write(struct.pack("<II", 1, len(arrays)))
        for a in arrays:
            a = np.ascontiguousarray(a)
            f.write(struct.pack("<II", DTYPES[a.dtype], a.ndim))
            f.write(struct.pack(f"<{a.ndim}q", *a.shape))
            f.write(a.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--pool_n", type=int, default=2000, help="층화 추출 모집단 크기 (eval_vrsbench --n과 동일)")
    ap.add_argument("--prompt", default="format_rules", choices=list(PROMPTS))
    ap.add_argument("--hf_ref", action="store_true", help="HF causal(bs=1, greedy) 참조 답변도 생성")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.load(open(join(ROOT, "VRSBench_EVAL_vqa.json")))
    data = stratified_sample(data, args.pool_n, 0)[:args.n]
    pre = join(args.out, "pre")
    os.makedirs(pre, exist_ok=True)

    processor = AutoProcessor.from_pretrained(CKPT, revision=REVISION, trust_remote_code=True)
    model = None
    if args.hf_ref:
        model = AutoModelForImageTextToText.from_pretrained(
            CKPT, revision=REVISION, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda").eval()

    requests, samples, t0 = [], [], time.time()
    for i, x in enumerate(data):
        qid = str(x["question_id"])
        src = join(ROOT, "images/Images_val", x["image_id"])
        link = join(pre, f"{qid}.png")
        if not os.path.lexists(link):
            os.symlink(src, link)
        text = x["question"] + PROMPTS[args.prompt]
        img = Image.open(src).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}, {"type": "image", "image": img}]}]
        inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)
        ids = inputs["input_ids"][0].numpy().astype(np.int32)
        pix = inputs["pixel_values"].numpy().astype(np.float16)
        pool = inputs["image_token_pooling"].numpy().astype(np.int64)
        write_sidecar(join(pre, f"{qid}.molmo2.bin"), [ids, pix, pool])

        rec = dict(qid=qid, image_id=x["image_id"], type=TYPE_MAP[x["type"]], raw_type=x["type"],
                   q=x["question"], gt=x["ground_truth"], n_in=len(ids), n_crops=pix.shape[0], n_patch=pool.shape[0])
        if model is not None:
            inputs.pop("token_type_ids")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            gen = out[0, len(ids):]
            rec["hf_ids"] = gen.tolist()
            rec["hf_pred"] = processor.decode(gen, skip_special_tokens=True).strip()
            rec["hf_em"], rec["hf_lenient"] = score(rec["hf_pred"], x["ground_truth"], rec["type"])
        samples.append(rec)
        requests.append({"messages": [{"role": "user", "content": [
            {"type": "text", "text": text}, {"type": "image", "image": link}]}]})
        if i % 20 == 0:
            print(f"{i + 1}/{len(data)} {time.time() - t0:.0f}s", flush=True)

    req = {"batch_size": 1, "temperature": 1.0, "top_p": 1.0, "top_k": 1,
           "max_generate_length": args.max_new_tokens, "apply_chat_template": False, "requests": requests}
    json.dump(req, open(join(args.out, "requests.json"), "w"), ensure_ascii=False, indent=1)
    with open(join(args.out, "samples.jsonl"), "w") as f:
        for r in samples:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("done", len(samples), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
