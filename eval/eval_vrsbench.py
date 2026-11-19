"""VRSBench VQA zero-shot 평가 (Molmo2-4B 원본 HF 가중치, greedy, 규칙 기반 채점).

usage: python eval_vrsbench.py --n 2000 --out results/vrsbench_zs_2000.jsonl
"""
import argparse
import collections
import json
import random
import re
import time
from os.path import join

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = "/home/user/ondeviceAI/molmo2/data/VRSBench"
CKPT = "allenai/Molmo2-4B"
REVISION = "042abfa7a38879a376cec03d949eff0aefaa0600"
PROMPTS = {
    "word_or_number": "\nAnswer the question using a single word or number.",
    "short": "\nAnswer the question with a short answer (yes/no, a number, or a short phrase).",
    "format_rules": "\nAnswer with as few words as possible. If the question asks how many, answer with a "
                    "number only. If it is a yes/no question, answer yes or no only. Otherwise answer "
                    "with a single word or short phrase.",
}

# 원시 12개 유형 -> 방법론 문서의 10개 유형
TYPE_MAP = {
    "object existence": "presence", "object quantity": "quantity", "object position": "position",
    "object category": "category", "object color": "color", "object shape": "shape",
    "object size": "size", "reasoning": "reasoning", "object direction": "direction",
    "scene type": "scene", "image": "scene", "rural or urban": "scene",
}
NUM_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
NUM_WORDS["none"] = "0"


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"<points[^>]*>(.*?)</points>", r"\1", s)
    s = re.sub(r"[^\w\s-]", " ", s)
    toks = [NUM_WORDS.get(t, t) for t in s.split() if t not in ("a", "an", "the")]
    return " ".join(toks)


def extract_count(pred):
    """Molmo2가 pointing으로 답한 경우 'total of N' 또는 point 개수를 사용."""
    m = re.search(r"total of (\d+)", pred)
    if m:
        return m.group(1)
    m = re.search(r'coords="([^"]*)"', pred)
    if m:
        nums = m.group(1).split()
        return str(len(nums) // 3) if nums else None
    return None


def score(pred, gt, qtype):
    p, g = normalize(pred), normalize(gt)
    if qtype == "quantity" and g.isdigit():
        c = extract_count(pred)
        if c is not None:
            p = c
    exact = p == g
    lenient = exact or (bool(g) and re.search(rf"\b{re.escape(g)}\b", p) is not None) or \
        (bool(p) and re.search(rf"\b{re.escape(p)}\b", g) is not None)
    return exact, lenient


def stratified_sample(data, n, seed):
    rng = random.Random(seed)
    by_t = collections.defaultdict(list)
    for x in data:
        by_t[TYPE_MAP[x["type"]]].append(x)
    out = []
    for t, xs in by_t.items():
        k = max(1, round(n * len(xs) / len(data)))
        out += rng.sample(xs, min(k, len(xs)))
    rng.shuffle(out)
    return out


def use_trt_visual(model, engine_path):
    """vision_backbone.forward를 TensorRT 엔진 호출로 교체 (배치 내 샘플별로 실행).

    엔진 출력은 <im_patch> 토큰 임베딩이 더해진 값이므로, HF가 다시 더하기 전에 빼 둔다.
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    engine = trt.Runtime(logger).deserialize_cuda_engine(open(engine_path, "rb").read())
    ctx = engine.create_execution_context()
    to_torch = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16, trt.DataType.INT64: torch.int64}
    img_dtype = to_torch[engine.get_tensor_dtype("images")]
    out_dtype = to_torch[engine.get_tensor_dtype("patch_embeds")]
    vb = model.model.vision_backbone
    wte = model.model.transformer.wte
    patch_embed = wte.new_embedding[model.config.image_patch_id - wte.embedding.shape[0]].detach().float()
    stream = torch.cuda.current_stream()

    def forward(images, pooled_patches_idx):
        outs = []
        for b in range(images.shape[0]):
            idx = pooled_patches_idx[b]
            idx = idx[(idx >= 0).any(-1)][None].to(torch.int64).contiguous()
            img = images[b:b + 1].to(device="cuda", dtype=img_dtype).contiguous()
            out = torch.empty((idx.shape[1], patch_embed.shape[0]), dtype=out_dtype, device="cuda")
            ctx.set_input_shape("images", tuple(img.shape))
            ctx.set_input_shape("pooled_patches_idx", tuple(idx.shape))
            ctx.set_tensor_address("images", img.data_ptr())
            ctx.set_tensor_address("pooled_patches_idx", idx.data_ptr())
            ctx.set_tensor_address("patch_embeds", out.data_ptr())
            ctx.execute_async_v3(stream.cuda_stream)
            outs.append(out.float() - patch_embed)
        return torch.cat(outs).to(vb.dtype)

    vb.forward = forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="0 = 전체 test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--prompt", default="word_or_number", choices=list(PROMPTS))
    ap.add_argument("--causal", action="store_true",
                    help="token_type_ids를 빼서 영상 블록 양방향 attention을 끄고 순수 causal로 실행 (Edge-LLM/Orin 조건)")
    ap.add_argument("--trt_visual", default=None,
                    help="시각부를 이 TensorRT 엔진(export_molmo2_visual.py 형식)으로 대체")
    ap.add_argument("--lora", default=None, help="LoRA adapter 디렉터리 (불러와 병합 후 평가)")
    ap.add_argument("--model_dir", default=None, help="로컬 체크포인트 (예: LoRA 병합본). 없으면 HF 원본")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.load(open(join(ROOT, "VRSBench_EVAL_vqa.json")))
    if args.n:
        data = stratified_sample(data, args.n, args.seed)

    ckpt, rev = (args.model_dir, None) if args.model_dir else (CKPT, REVISION)
    model = AutoModelForImageTextToText.from_pretrained(
        ckpt, revision=rev, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda").eval()
    processor = AutoProcessor.from_pretrained(
        ckpt, revision=rev, trust_remote_code=True, padding_side="left")
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora).merge_and_unload().eval()
    if args.trt_visual:
        use_trt_visual(model, args.trt_visual)

    results, t0 = [], time.time()
    for i in range(0, len(data), args.bs):
        batch = data[i:i + args.bs]
        msgs = [[{"role": "user", "content": [
            {"type": "text", "text": x["question"] + PROMPTS[args.prompt]},
            {"type": "image", "image": Image.open(join(ROOT, "images/Images_val", x["image_id"])).convert("RGB")},
        ]}] for x in batch]
        inputs = processor.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True)
        n_in = inputs["input_ids"].shape[1]
        if args.causal:
            inputs.pop("token_type_ids")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        preds = processor.batch_decode(out[:, n_in:], skip_special_tokens=True)
        for x, p in zip(batch, preds):
            t = TYPE_MAP[x["type"]]
            em, len_ = score(p, x["ground_truth"], t)
            results.append(dict(qid=x["question_id"], type=t, raw_type=x["type"], q=x["question"],
                                gt=x["ground_truth"], pred=p.strip(), em=em, lenient=len_))
        if (i // args.bs) % 20 == 0:
            acc = sum(r["em"] for r in results) / len(results)
            print(f"{len(results)}/{len(data)} em={acc:.3f} {time.time() - t0:.0f}s", flush=True)

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    agg = collections.defaultdict(lambda: [0, 0, 0])
    for r in results:
        for k in (r["type"], "ALL"):
            agg[k][0] += 1
            agg[k][1] += r["em"]
            agg[k][2] += r["lenient"]
    print(f"\n{'type':<10}{'n':>7}{'EM%':>8}{'lenient%':>10}")
    for k, (n, em, le) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"{k:<10}{n:>7}{100 * em / n:>8.1f}{100 * le / n:>10.1f}")
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
