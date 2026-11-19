"""Molmo2 출력 어휘 축소용 vocab_map 생성 (VRSBench train 기준, Orin 메모리 절감 시험 V2, 2026-11-16).

Edge-LLM 기본 도구(tensorrt_edgellm/scripts/reduce_vocab.py)는 cnn_dailymail로 고정돼 있어 직접 만듦.
lm_head 출력 후보만 줄이고 입력 embedding은 그대로임(질문 단어는 전체 어휘로 읽음).

선택 순서 (test 데이터는 쓰지 않음)
  1. 특수 토큰: EOS·BOS·PAD (<|im_end|> 151645, <|endoftext|> 151643)
  2. VRSBench train(LoRA 검증 이미지 제외) 정답 토큰 전부
  3. train 질문 토큰을 빈도순으로
  4. 남는 자리는 낮은 id 순(BPE 병합 순서상 흔한 조각)으로 채워 목표 크기를 맞춤

usage (molmo2/.venv):
  molmo2/.venv/bin/python quant/scripts/make_vocab_map.py --size 16384 --out quant/vocab/molmo2_v16k
"""
import argparse
import collections
import json
import os
import sys
from os.path import join

ROOT = "/home/user/ondeviceAI"
sys.path.insert(0, join(ROOT, "lora/scripts"))
from train_lora_vrsbench import load_vqa  # noqa: E402

TOK = join(ROOT, "edgellm_check/ws_lora/causal_r16_ep1/hf")
VOCAB = 151936


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16384)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK)

    train, _ = load_vqa(seed=0, val_frac=0.01)
    ans, qs = collections.Counter(), collections.Counter()
    for r in train:
        ans.update(tok.encode(r["a"], add_special_tokens=False))
        qs.update(tok.encode(r["q"], add_special_tokens=False))

    special = {151645, 151643}
    for t in (tok.eos_token_id, tok.bos_token_id, tok.pad_token_id):
        if t is not None:
            special.add(int(t))
    sel = set(special) | set(ans)
    n_ans = len(set(ans))
    for t, _ in qs.most_common():
        if len(sel) >= a.size:
            break
        sel.add(int(t))
    n_after_q = len(sel)
    t = 0
    while len(sel) < a.size:
        sel.add(t)
        t += 1
    sel = sorted(x for x in sel if 0 <= x < VOCAB)[:a.size]
    assert len(sel) == a.size and special <= set(sel)

    os.makedirs(a.out, exist_ok=True)
    save_file({"vocab_map": torch.tensor(sel, dtype=torch.int32)}, join(a.out, "vocab_map.safetensors"))
    info = dict(vocab_size=VOCAB, reduced_vocab_size=a.size, method="vrsbench_train_answers_questions_lowid",
                dataset="VRSBench train (load_vqa seed 0, val 1% 제외)", n_train=len(train),
                answer_unique_tokens=n_ans, after_question_tokens=n_after_q, special=sorted(special))
    json.dump(info, open(join(a.out, "reduced_vocab.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
