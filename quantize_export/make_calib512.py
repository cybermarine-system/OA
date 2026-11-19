"""AWQ 보정 텍스트 512개 생성 (VRSBench train, LoRA 검증 이미지 제외, 질문 유형별 층화 추출, seed 0).

- 학습 스크립트(lora/scripts/train_lora_vrsbench.py)의 load_vqa(seed=0, val_frac=0.01)로 학습/검증 분할을 그대로 재현하고
  학습 분할에서만 뽑음. test(VRSBench_EVAL_vqa)는 train과 이미지가 겹치지 않음
- 질문 유형은 VRSBench_train.json에 없어서 ann/Annotations_train/<이미지>.json의 qa_pairs와 (이미지, 질문)으로 맞춰 붙임
  유형은 평가와 같은 10개 유형(TYPE_MAP)으로 묶고, 비례 배분(최대 나머지 방식)으로 정확히 512개
- 텍스트: 질문(clean_question) + format_rules 지시문을 모델별 채팅 템플릿(이미지 없음, add_generation_prompt=True)으로 감싸고
  정답 + <|im_end|>를 이어 붙임 (학습 때 손실을 건 부분과 같은 형식)
- 출력: quant/calib/vrsbench_calib512.jsonl (한 줄에 문항 정보 + text_molmo2 / text_qwen3vl4b / text_qwen3vl2b)

usage (molmo2/.venv):
  molmo2/.venv/bin/python quant/scripts/make_calib512.py
"""
import collections
import json
import random
import re
import sys
from os.path import join

ROOT = "/home/user/ondeviceAI"
sys.path.insert(0, join(ROOT, "lora/scripts"))
sys.path.insert(0, join(ROOT, "molmo2/scripts"))
from train_lora_vrsbench import VRS, load_vqa  # noqa: E402
from eval_vrsbench import PROMPTS, TYPE_MAP  # noqa: E402

N, SEED = 512, 0
OUT = join(ROOT, "quant/calib/vrsbench_calib512.jsonl")
TOKENIZERS = {
    "molmo2": join(ROOT, "edgellm_check/ws_lora/causal_r16_ep1/hf"),
    "qwen3vl4b": join(ROOT, "edgellm_check/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"),
    "qwen3vl2b": join(ROOT, "edgellm_check/hf_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"),
}


def norm(s):
    return re.sub(r"[\s.?]+$", "", s.strip().lower())


def attach_types(rows):
    cache, miss = {}, 0
    for r in rows:
        stem = r["image"].rsplit(".", 1)[0]
        if stem not in cache:
            qa = json.load(open(join(VRS, "ann/Annotations_train", stem + ".json")))["qa_pairs"]
            cache[stem] = {norm(q["question"]): q["type"] for q in qa}
        t = cache[stem].get(norm(r["q"]))
        if t is None:
            miss += 1
        r["raw_type"] = t
        r["type"] = TYPE_MAP.get(t) if t else None
    return miss


def main():
    train, val = load_vqa(seed=0, val_frac=0.01)
    val_imgs = {r["image"] for r in val}
    print(f"train {len(train)} / val {len(val)} (val 이미지 {len(val_imgs)})")
    miss = attach_types(train)
    typed = [r for r in train if r["type"]]
    print(f"유형 매칭 실패 {miss}건 (제외)")
    by_t = collections.defaultdict(list)
    for r in typed:
        by_t[r["type"]].append(r)
    # 비례 배분, 최대 나머지 방식으로 합계 512
    quota = {t: N * len(xs) / len(typed) for t, xs in by_t.items()}
    k = {t: int(q) for t, q in quota.items()}
    for t in sorted(quota, key=lambda t: -(quota[t] - k[t]))[:N - sum(k.values())]:
        k[t] += 1
    rng = random.Random(SEED)
    out = []
    for t in sorted(by_t):
        out += rng.sample(by_t[t], k[t])
    rng.shuffle(out)
    assert len(out) == N and not ({r["image"] for r in out} & val_imgs)

    from transformers import AutoTokenizer
    toks = {m: AutoTokenizer.from_pretrained(p) for m, p in TOKENIZERS.items()}
    prompt = PROMPTS["format_rules"]
    with open(OUT, "w") as f:
        for i, r in enumerate(out):
            rec = dict(idx=i, image=r["image"], type=r["type"], raw_type=r["raw_type"], q=r["q"], a=r["a"])
            msgs = [{"role": "user", "content": r["q"] + prompt}]
            for m, tok in toks.items():
                pre = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                rec["text_" + m] = pre + r["a"] + "<|im_end|>"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    cnt = collections.Counter(r["type"] for r in out)
    print("유형별 개수:", dict(sorted(cnt.items(), key=lambda kv: -kv[1])))
    print("모집단 비율:", {t: f"{len(xs) / len(typed):.3f}" for t, xs in by_t.items()})
    for m, tok in toks.items():
        lens = [len(tok(json.loads(l)["text_" + m])["input_ids"]) for l in open(OUT)]
        print(f"{m}: 토큰 길이 평균 {sum(lens) / len(lens):.1f}, 최대 {max(lens)}")
    print("예시 molmo2:", repr(json.loads(open(OUT).readline())["text_molmo2"]))
    print("예시 qwen3vl4b:", repr(json.loads(open(OUT).readline())["text_qwen3vl4b"]))
    print("저장", OUT)


if __name__ == "__main__":
    main()
