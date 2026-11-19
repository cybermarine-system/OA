"""Qwen3-VL을 CountBench / PixMo-Count test로 평가 (HF, BF16, greedy, bs=1).

Molmo2 원본 평가와 같은 데이터(원본 repo가 만든 로컬 데이터셋)를 쓰고, Qwen은 point를 출력하지 않으므로
답에 적힌 숫자로 채점한다. 같은 기준 비교를 위해 Molmo2 원본 eval 예측도 숫자 기준으로 다시 채점한다.

usage: python eval_counting_qwen3vl.py --model <HF snapshot> --name qwen3vl4b
       python eval_counting_qwen3vl.py --rescore_molmo2
"""
import argparse
import glob
import json
import re
from os.path import join

ROOT = "/home/user/ondeviceAI"
DATA = join(ROOT, "molmo2/data/molmo/torch_datasets")
RESULTS = join(ROOT, "molmo2/results")
SUFFIX = " Answer with a number only."
WORD_TO_NUM = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
WORD_TO_NUM.update(none=0, no=0)


def parse_count(text):
    t = text.lower()
    m = re.search(r"total of (\d+)", t)
    if m:
        return int(m.group(1))
    m = re.search(r"\d+", t)
    if m:
        return int(m.group(0))
    for w in re.findall(r"[a-z]+", t):
        if w in WORD_TO_NUM:
            return WORD_TO_NUM[w]
    return None


def summarize(rows):
    n = len(rows)
    correct = sum(r["pred"] == r["gt"] for r in rows) / n * 100
    close = sum(r["pred"] is not None and abs(r["pred"] - r["gt"]) <= 1 for r in rows) / n * 100
    valid = sum(r["pred"] is not None for r in rows) / n * 100
    return dict(n=n, correct=round(correct, 2), close=round(close, 2), valid=round(valid, 2))


def load_tasks():
    import datasets

    cb = datasets.load_from_disk(join(DATA, "academic_datasets/countbench_qa"))
    pc = datasets.load_from_disk(join(DATA, "pixmo_datasets/count"))["test"]
    tasks = {
        "countbench": [dict(id=ex["example_id"], image=ex["image"], question=ex["question"], gt=ex["count"])
                       for ex in cb],
        "pixmo_count": [dict(id=ex["image_url"], image=ex["image"],
                             question=f"How many {ex['label']} are in the image?", gt=ex["count"])
                        for ex in pc],
    }
    return tasks


def run_qwen(model_dir, name):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda").eval()
    processor = AutoProcessor.from_pretrained(model_dir)
    summary = {}
    for task, items in load_tasks().items():
        rows = []
        for it in items:
            img = it["image"] if not isinstance(it["image"], str) else Image.open(it["image"])
            msgs = [{"role": "user", "content": [{"type": "image", "image": img.convert("RGB")},
                                                 {"type": "text", "text": it["question"] + SUFFIX}]}]
            inputs = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                   return_dict=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
            text = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            rows.append(dict(id=it["id"], gt=it["gt"], pred=parse_count(text), text=text))
        with open(join(RESULTS, f"counting_{task}_{name}_hf.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary[task] = summarize(rows)
        print(name, task, summary[task], flush=True)
    return summary


def rescore_molmo2():
    summary = {}
    for task, pattern in [("countbench", "native_countbench/*/predictions.json"),
                          ("pixmo_count", "native_pixmo_count/*/predictions.json")]:
        preds = json.load(open(glob.glob(join(RESULTS, pattern))[0]))
        rows = [dict(gt=p["count"], pred=parse_count(re.sub(r"<points[^>]*>", "", p["prediction"])))
                for p in preds]
        summary[task] = summarize(rows)
        print("molmo2 (숫자 기준 재채점)", task, summary[task])
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--name")
    ap.add_argument("--rescore_molmo2", action="store_true")
    args = ap.parse_args()
    if args.rescore_molmo2:
        rescore_molmo2()
    else:
        run_qwen(args.model, args.name)
