"""같은 문항 집합에 대한 두 평가 결과(jsonl)를 문항 단위로 비교: 정확도 차이, 정답↔오답 전환 수, McNemar 검정.

usage: python compare_runs.py <base.jsonl> <other.jsonl> [<other2.jsonl> ...]
"""
import collections
import json
import math
import sys


def load(path):
    return {r["qid"]: r for r in map(json.loads, open(path))}


def mcnemar_p(b, c):
    """연속성 보정 카이제곱 McNemar (자유도 1) p-value."""
    if b + c == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return math.erfc(math.sqrt(chi2 / 2))


base = load(sys.argv[1])
for path in sys.argv[2:]:
    other = load(path)
    assert base.keys() == other.keys(), "문항 집합이 다름"
    print(f"\n== {path.split('/')[-1]} vs {sys.argv[1].split('/')[-1]}")
    print(f"{'type':<10}{'n':>6}{'base EM':>9}{'EM':>7}{'Δ%p':>7}{'O→X':>6}{'X→O':>6}{'p':>8}")
    groups = collections.defaultdict(list)
    for q in base:
        groups["ALL"].append(q)
        groups[base[q]["type"]].append(q)
    for t, qs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        a = sum(base[q]["em"] for q in qs) / len(qs) * 100
        o = sum(other[q]["em"] for q in qs) / len(qs) * 100
        b = sum(base[q]["em"] and not other[q]["em"] for q in qs)
        c = sum(not base[q]["em"] and other[q]["em"] for q in qs)
        print(f"{t:<10}{len(qs):>6}{a:>9.1f}{o:>7.1f}{o - a:>+7.1f}{b:>6}{c:>6}{mcnemar_p(b, c):>8.3f}")
