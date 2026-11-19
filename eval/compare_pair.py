"""같은 문항에 대한 두 결과(jsonl: qid, type, pred, em)를 대응 비교.

- 정확도 차이(B−A, %p)와 대응표본 95% 신뢰구간(Wald, 대응 차이의 분산 사용 + Newcombe 방식 아님, 근사)
- 응답 일치율(pred 문자열 완전 일치, 앞뒤 공백 제거)
- McNemar 정확검정 (b: A정답→B오답, c: A오답→B정답)
- 유형별 같은 지표 + 유형별 Holm 보정 p
- --json으로 수치 저장

usage: molmo2/.venv/bin/python quant/scripts/compare_pair.py A.jsonl B.jsonl [--names FP16 INT4] [--json out.json]
"""
import argparse
import collections
import json
import math


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def stats(pairs):
    n = len(pairs)
    a = sum(x for x, _, _ in pairs)
    b_ = sum(y for _, y, _ in pairs)
    b = sum(1 for x, y, _ in pairs if x and not y)
    c = sum(1 for x, y, _ in pairs if y and not x)
    same = sum(s for _, _, s in pairs)
    d = (b_ - a) / n
    # 대응 차이 d_i ∈ {-1,0,1}의 표본분산으로 Wald CI
    var = ((b + c) / n - d * d) / n
    hw = 1.96 * math.sqrt(max(var, 0))
    return dict(n=n, acc_a=100 * a / n, acc_b=100 * b_ / n, diff_pp=100 * d, ci_lo=100 * (d - hw),
                ci_hi=100 * (d + hw), b=b, c=c, p=mcnemar_exact(b, c), agree=100 * same / n)


def load(p):
    return {str(r["qid"]): r for r in map(json.loads, open(p))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--json")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    keys = [k for k in A if k in B]
    if len(keys) != len(A) or len(keys) != len(B):
        print(f"주의: 공통 문항 {len(keys)} (A {len(A)}, B {len(B)})")
    rows = collections.defaultdict(list)
    for k in keys:
        x, y = A[k], B[k]
        t = (bool(x["em"]), bool(y["em"]), x["pred"].strip() == y["pred"].strip())
        rows["ALL"].append(t)
        rows[x["type"]].append(t)
    res = {t: stats(v) for t, v in rows.items()}
    types = [t for t in res if t != "ALL"]
    for t, p in zip(types, holm([res[t]["p"] for t in types])):
        res[t]["p_holm"] = p
    na, nb = args.names
    print(f"{'유형':<10}{'n':>6}{na:>8}{nb:>8}{'차이%p':>8}{'95% CI':>17}{'b/c':>9}{'p':>8}{'Holm p':>8}{'일치%':>7}")
    for t in ["ALL"] + sorted(types, key=lambda t: -res[t]["n"]):
        s = res[t]
        print(f"{t:<10}{s['n']:>6}{s['acc_a']:>8.1f}{s['acc_b']:>8.1f}{s['diff_pp']:>+8.2f}"
              f"  [{s['ci_lo']:+6.2f},{s['ci_hi']:+6.2f}]{s['b']:>5}/{s['c']:<4}{s['p']:>7.3f}"
              f"{s.get('p_holm', float('nan')):>8.3f}{s['agree']:>7.1f}")
    if args.json:
        json.dump(dict(a=args.a, b=args.b, names=args.names, stats=res), open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
