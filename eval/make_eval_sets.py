"""Edge-LLM 평가·Orin 측정용 문항 집합과 llm_inference 요청 JSON 생성 (세 모델 공통 요청 형식).

요청 형식은 기존과 같음: messages = [user: text(질문 + format_rules) → image], apply_chat_template true, greedy(top_k 1).
Molmo2 runner(C++ 전처리 "cpp")와 Qwen3-VL 공식 경로가 같은 요청 파일을 그대로 읽음 (Molmo2는 이미지 경로만 사용)

만드는 것
1) A100 test 전체: quant/sets/test_full_requests.json (+ _samples.jsonl), 37,409문항, 원본 JSON 순서, 최대 생성 48토큰,
   이미지 경로는 작업 폴더 루트 기준 상대경로 → /home/user/ondeviceAI 에서 실행
2) Orin 공통 묶음 deploy/orin/common/ (이미지 경로는 묶음 루트 기준 images/<파일>, 이미지는 심볼릭 링크)
   - consistency2000: 기존 2,001 층화 표본(eval_vrsbench.stratified_sample(n=2000, seed 0))의 앞 2,000개.
     표본 순서가 이미 seed 0으로 섞여 있어 앞 2,000개도 무작위 부분집합. A100 결과를 qid로 그대로 재사용 가능. 최대 생성 48토큰 (A100과 같게)
   - efficiency500: test에서 2,001 표본의 질문·이미지를 모두 뺀 나머지에서 유형별 층화 500개(seed 0). 최대 생성 16토큰
   - warmup10: 위 두 집합의 질문·이미지와 겹치지 않는 10개(seed 0). 최대 생성 16토큰

usage: molmo2/.venv/bin/python quant/scripts/make_eval_sets.py
"""
import collections
import json
import os
import random
import sys
from os.path import join

R = "/home/user/ondeviceAI"
sys.path.insert(0, join(R, "molmo2/scripts"))
from eval_vrsbench import PROMPTS, ROOT, TYPE_MAP, stratified_sample  # noqa: E402

PROMPT = PROMPTS["format_rules"]
IMG_REL = os.path.relpath(join(ROOT, "images/Images_val"), R)


def strat_exact(data, n, seed):
    """비례 배분(최대 나머지)으로 정확히 n개."""
    by_t = collections.defaultdict(list)
    for x in data:
        by_t[TYPE_MAP[x["type"]]].append(x)
    q = {t: n * len(v) / len(data) for t, v in by_t.items()}
    k = {t: int(v) for t, v in q.items()}
    for t in sorted(q, key=lambda t: -(q[t] - k[t]))[:n - sum(k.values())]:
        k[t] += 1
    rng = random.Random(seed)
    out = []
    for t in sorted(by_t):
        out += rng.sample(by_t[t], k[t])
    rng.shuffle(out)
    return out


def write(items, req_path, img_prefix, max_new):
    reqs = [{"messages": [{"role": "user", "content": [
        {"type": "text", "text": x["question"] + PROMPT},
        {"type": "image", "image": f"{img_prefix}/{x['image_id']}"}]}]} for x in items]
    json.dump({"batch_size": 1, "temperature": 1.0, "top_p": 1.0, "top_k": 1, "max_generate_length": max_new,
               "apply_chat_template": True, "requests": reqs}, open(req_path, "w"), ensure_ascii=False, indent=1)
    with open(req_path.replace(".json", "_samples.jsonl"), "w") as f:
        for x in items:
            f.write(json.dumps(dict(qid=str(x["question_id"]), image_id=x["image_id"], type=TYPE_MAP[x["type"]],
                                    raw_type=x["type"], q=x["question"], gt=x["ground_truth"]),
                               ensure_ascii=False) + "\n")
    c = collections.Counter(TYPE_MAP[x["type"]] for x in items)
    print(f"{req_path}: {len(items)}문항, 이미지 {len({x['image_id'] for x in items})}장, {dict(c.most_common())}")


def main():
    data = json.load(open(join(ROOT, "VRSBench_EVAL_vqa.json")))
    os.makedirs(join(R, "quant/sets"), exist_ok=True)
    write(data, join(R, "quant/sets/test_full_requests.json"), IMG_REL, 48)

    s2001 = stratified_sample(data, 2000, 0)
    assert len(s2001) == 2001
    cons = s2001[:2000]
    used_q = {x["question_id"] for x in s2001}
    used_img = {x["image_id"] for x in s2001}
    rest = [x for x in data if x["question_id"] not in used_q and x["image_id"] not in used_img]
    eff = strat_exact(rest, 500, 0)
    used_img2 = used_img | {x["image_id"] for x in eff}
    rest2 = [x for x in rest if x["image_id"] not in used_img2]
    warm = random.Random(0).sample(rest2, 10)
    print(f"test {len(data)}, 2,001 표본 제외 후(질문·이미지) {len(rest)}")

    C = join(R, "deploy/orin/common")
    os.makedirs(join(C, "images"), exist_ok=True)
    write(cons, join(C, "consistency2000_requests.json"), "images", 48)
    write(eff, join(C, "efficiency500_requests.json"), "images", 16)
    write(warm, join(C, "warmup10_requests.json"), "images", 16)
    imgs = {x["image_id"] for x in cons + eff + warm}
    for im in imgs:
        dst = join(C, "images", im)
        if not os.path.lexists(dst):
            os.symlink(join(ROOT, "images/Images_val", im), dst)
    size = sum(os.path.getsize(join(ROOT, "images/Images_val", im)) for im in imgs)
    q = [set(x["question_id"] for x in s) for s in (cons, eff, warm)]
    i = [set(x["image_id"] for x in s) for s in (cons, eff, warm)]
    assert not (q[0] & q[1] or q[0] & q[2] or q[1] & q[2] or i[0] & i[1] or i[0] & i[2] or i[1] & i[2])
    json.dump(dict(consistency2000="2,001 층화 표본(seed 0)의 앞 2,000개", efficiency500="2,001 표본 질문·이미지 제외 후 유형별 층화 500(seed 0)",
                   warmup10="위 두 집합과 질문·이미지 모두 겹치지 않는 무작위 10개(seed 0)", images=len(imgs),
                   images_bytes=size, overlap="질문·이미지 모두 집합 간 겹침 0"),
              open(join(C, "sets_info.json"), "w"), ensure_ascii=False, indent=1)
    print(f"이미지 {len(imgs)}장 {size / 1e9:.3f} GB (심볼릭 링크)")


if __name__ == "__main__":
    main()
