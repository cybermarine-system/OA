"""조건 ①~⑤ 비교 그래프, 발표용
- lora/figures/conditions_em_20261108.png : 전체 EM 막대
- lora/figures/conditions_diff_20261108.png : ③ 대비 차이와 95% 신뢰구간 (1 epoch)
"""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import koreanize_matplotlib  # noqa: F401

R = "molmo2/results/"; L = "vrsbench_2001_rules_molmo2_lora_"
F = {"①": "vrsbench_zs_2000_rules.jsonl", "②": "vrsbench_zs_2000_rules_causal.jsonl",
     "③": L + "bidir_r16_bidir.jsonl", "④": L + "bidir_r16_causal.jsonl", "⑤": L + "causal_r16_causal.jsonl"}
D = {k: {r["qid"]: r["em"] for r in map(json.loads, open(R + v))} for k, v in F.items()}
em = {k: sum(v.values()) / len(v) * 100 for k, v in D.items()}
GRAY, ORG, BLUE = "#9a9a9a", "#eb6834", "#2a78d6"

# 1) 전체 EM 막대
labels = ["①\n미학습\n양방향 실행", "②\n미학습\ncausal 실행", "③\n양방향 학습\n양방향 실행", "④\n양방향 학습\ncausal 실행", "⑤\ncausal 학습\ncausal 실행"]
cols = [GRAY, GRAY, ORG, ORG, BLUE]
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
bars = ax.bar(range(5), [em[k] for k in F], color=cols, width=0.6)
for i, k in enumerate(F):
    ax.text(i, em[k] + 1, f"{em[k]:.1f}", ha="center", va="bottom", fontsize=12)
ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 80); ax.set_ylabel("정답률 EM (%)", fontsize=11)
ax.grid(axis="y", color="#e6e6e6"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("lora/figures/conditions_em_20261108.png")

# 2) ③(원래 방식) 대비 차이, 95% 신뢰구간
def diff(a, b):
    d = [D[b][q] - D[a][q] for q in D[a]]; n = len(d); m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n)
    return m * 100, 1.96 * se * 100
rows = [("④ 양방향 학습 · causal 실행", "④", ORG), ("⑤ causal 학습 · causal 실행", "⑤", BLUE)]
fig, ax = plt.subplots(figsize=(8, 3.0), dpi=200)
for i, (lab, k, c) in enumerate(rows):
    m, h = diff("③", k)
    y = 1 - i
    ax.errorbar(m, y, xerr=h, fmt="o", color=c, ms=9, capsize=6, lw=2)
    ax.text(m, y + 0.22, f"{m:+.2f}%p", ha="center", fontsize=11)
ax.axvline(0, color="#555555", lw=1)
ax.set_yticks([1, 0]); ax.set_yticklabels([r[0] for r in rows], fontsize=11)
ax.set_ylim(-0.6, 1.6); ax.set_xlim(-3, 1)
ax.set_xlabel("③ 양방향 학습 · 양방향 실행 대비 정답률 차이 (%p, 95% 신뢰구간)", fontsize=10.5)
ax.grid(axis="x", color="#e6e6e6"); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout(); fig.savefig("lora/figures/conditions_diff_20261108.png")
print({k: round(v, 2) for k, v in em.items()}, diff("③", "④"), diff("③", "⑤"))
