"""Edge-LLM Molmo2 runner 결과와 HF causal 참조 결과를 같은 문항 기준으로 비교.

입력: make_molmo2_inputs.py --hf_ref가 만든 samples.jsonl, llm_inference --outputFile 결과 JSON
출력: 유형별 EM(HF / Edge-LLM), 답변 문자열 완전 일치율, McNemar 정확검정, 불일치 사례. --out으로 jsonl 저장.
선택: --batch_ref에 eval_vrsbench.py 결과(jsonl, 배치 16 left-padding)를 주면 같은 qid의 EM도 함께 표시.

usage: molmo2/.venv/bin/python compare_edgellm_hf.py <samples.jsonl> <edgellm_out.json> [--batch_ref <jsonl>] [--out <jsonl>]
"""
import argparse
import collections
import json
import math
import sys

sys.path.insert(0, "/home/user/ondeviceAI/molmo2/scripts")
from eval_vrsbench import score  # noqa: E402


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("edgellm_out")
    ap.add_argument("--batch_ref")
    ap.add_argument("--out")
    args = ap.parse_args()

    samples = [json.loads(l) for l in open(args.samples)]
    resp = json.load(open(args.edgellm_out))["responses"]
    assert len(resp) == len(samples), (len(resp), len(samples))
    batch = {}
    if args.batch_ref:
        batch = {str(json.loads(l)["qid"]): json.loads(l) for l in open(args.batch_ref)}

    rows = []
    for x, r in zip(samples, resp):
        el = r["output_text"].strip()
        em, len_ = score(el, x["gt"], x["type"])
        rows.append(dict(x, el_pred=el, el_em=em, el_lenient=len_, el_tokens=r["generated_token_count"],
                         same_text=(el == x["hf_pred"]), batch_em=batch.get(x["qid"], {}).get("em")))

    agg = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        for k in (r["type"], "ALL"):
            a = agg[k]
            a["n"] += 1
            a["hf"] += r["hf_em"]
            a["el"] += r["el_em"]
            a["same"] += r["same_text"]
            a["b"] += r["hf_em"] and not r["el_em"]
            a["c"] += r["el_em"] and not r["hf_em"]
            if r["batch_em"] is not None:
                a["nb"] += 1
                a["batch"] += r["batch_em"]
    print(f"{'type':<10}{'n':>5}{'HF EM':>8}{'EL EM':>8}{'diff':>7}{'same%':>8}{'b/c':>7}{'p':>7}"
          + (f"{'batchEM':>9}" if batch else ""))
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["n"]):
        n = a["n"]
        line = (f"{k:<10}{n:>5}{100 * a['hf'] / n:>8.1f}{100 * a['el'] / n:>8.1f}"
                f"{100 * (a['el'] - a['hf']) / n:>+7.1f}{100 * a['same'] / n:>8.1f}"
                f"{a['b']:>4}/{a['c']:<2}{mcnemar_exact(a['b'], a['c']):>7.3f}")
        if batch:
            line += f"{100 * a['batch'] / a['nb']:>9.1f}" if a["nb"] else f"{'-':>9}"
        print(line)
    diff = [r for r in rows if not r["same_text"]]
    print(f"\n답변 불일치 {len(diff)}건 (최대 15건 표시)")
    for r in diff[:15]:
        print(f"  {r['qid']} [{r['type']}] gt={r['gt']!r} HF={r['hf_pred']!r} EL={r['el_pred']!r}")
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
