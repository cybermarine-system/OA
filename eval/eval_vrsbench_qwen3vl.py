"""VRSBench VQA 평가 — Qwen3-VL (HF Transformers 기준선) + 같은 문항의 Edge-LLM 요청 JSON 생성.

molmo2/scripts/eval_vrsbench.py와 같은 층화 추출(seed 0, n=2000 → 2,001쌍)·프롬프트·채점을 쓰고,
Edge-LLM 비교를 위해 batch 1, greedy, 최대 48토큰으로 실행한다. 메시지 구성도 같다 (text → image 순서).

usage (molmo2/.venv, HF_HOME=edgellm_check/hf_cache):
  python eval_vrsbench_qwen3vl.py --model Qwen/Qwen3-VL-4B-Instruct --out molmo2/results/vrsbench_2001_rules_qwen3vl4b_hf.jsonl \
      [--requests_out edgellm_check/ws_qwen/vrs2001_requests.json] [--n 2000] [--limit 0]
  --requests_out만 만들고 HF 평가를 건너뛰려면 --skip_hf
  --lora <adapter 폴더>: LoRA adapter를 얹어 평가 (2026-11-08 추가)
  --n 0: 층화 추출 없이 test 전체 37,409문항 (2026-11-08 추가, 순서는 원본 JSON 순서)
"""
import argparse
import collections
import json
import sys
import time
from os.path import join

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, "/home/user/ondeviceAI/molmo2/scripts")
from eval_vrsbench import PROMPTS, ROOT, TYPE_MAP, score, stratified_sample  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=2000, help="층화 추출 크기 (eval_vrsbench.py --n과 동일)")
    ap.add_argument("--limit", type=int, default=0, help="앞쪽 일부만 (0=전체)")
    ap.add_argument("--prompt", default="format_rules", choices=list(PROMPTS))
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--out")
    ap.add_argument("--requests_out")
    ap.add_argument("--skip_hf", action="store_true")
    ap.add_argument("--lora", default=None, help="PEFT adapter 폴더 (선택)")
    args = ap.parse_args()

    data = json.load(open(join(ROOT, "VRSBench_EVAL_vqa.json")))
    if args.n > 0:
        data = stratified_sample(data, args.n, 0)
    if args.limit:
        data = data[:args.limit]

    if args.requests_out:
        reqs = [{"messages": [{"role": "user", "content": [
            {"type": "text", "text": x["question"] + PROMPTS[args.prompt]},
            {"type": "image", "image": join(ROOT, "images/Images_val", x["image_id"])}]}]} for x in data]
        json.dump({"batch_size": 1, "temperature": 1.0, "top_p": 1.0, "top_k": 1,
                   "max_generate_length": args.max_new_tokens, "apply_chat_template": True, "requests": reqs},
                  open(args.requests_out, "w"), ensure_ascii=False, indent=1)
        meta = args.requests_out.replace(".json", "_samples.jsonl")
        with open(meta, "w") as f:
            for x in data:
                f.write(json.dumps(dict(qid=str(x["question_id"]), image_id=x["image_id"], type=TYPE_MAP[x["type"]],
                                        raw_type=x["type"], q=x["question"], gt=x["ground_truth"]),
                                   ensure_ascii=False) + "\n")
        print("requests", len(reqs), args.requests_out)
    if args.skip_hf:
        return

    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    if args.lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora).eval()
    processor = AutoProcessor.from_pretrained(args.model)

    results, t0 = [], time.time()
    for i, x in enumerate(data):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": x["question"] + PROMPTS[args.prompt]},
            {"type": "image", "image": Image.open(join(ROOT, "images/Images_val", x["image_id"])).convert("RGB")},
        ]}]
        inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True).to("cuda")
        n_in = inputs["input_ids"].shape[1]
        n_img = int((inputs["input_ids"] == model.config.image_token_id).sum())
        torch.cuda.synchronize()
        ts = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
        dt = (time.time() - ts) * 1000
        gen = out[0, n_in:]
        pred = processor.decode(gen, skip_special_tokens=True).strip()
        t = TYPE_MAP[x["type"]]
        em, len_ = score(pred, x["ground_truth"], t)
        results.append(dict(qid=str(x["question_id"]), type=t, raw_type=x["type"], q=x["question"],
                            gt=x["ground_truth"], pred=pred, em=em, lenient=len_, n_in=n_in, n_img_tokens=n_img,
                            n_gen=len(gen), hf_ms=dt))
        if i % 200 == 0:
            print(f"{i + 1}/{len(data)} em={sum(r['em'] for r in results) / len(results):.3f} "
                  f"{time.time() - t0:.0f}s n_in={n_in} n_img={n_img}", flush=True)

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
