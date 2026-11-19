"""유형별 정답률: 학습 전(causal 실행, ②) vs causal LoRA 1 epoch(⑤), 발표용
(lora/figures/type_em_causal_20261108.png)"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import koreanize_matplotlib  # noqa: F401

R = "molmo2/results/"
A = R + "vrsbench_zs_2000_rules_causal.jsonl"
B = R + "vrsbench_2001_rules_molmo2_lora_causal_r16_causal.jsonl"
T = [("전체", None), ("존재", "presence"), ("수량", "quantity"), ("위치", "position"), ("범주", "category"),
     ("장면", "scene"), ("색", "color"), ("형태", "shape"), ("크기", "size"), ("추론", "reasoning"), ("방향", "direction")]
def em(path):
    rs = [json.loads(l) for l in open(path)]
    out = []
    for _, t in T:
        q = [r for r in rs if t is None or r["type"] == t]
        out.append(round(sum(r["em"] for r in q) / len(q) * 100, 1))
    return out
a, b = em(A), em(B)
print(a); print(b)
fig, ax = plt.subplots(figsize=(9.4, 6.0), dpi=200)
w = 0.36; g = 0.2; xs = range(len(T))
ax.bar([x - g for x in xs], a, w, color="#BFBFBF", label="학습 전")
ax.bar([x + g for x in xs], b, w, color="#2F5597", label="LoRA 1 epoch 학습 후")
for x, (va, vb) in enumerate(zip(a, b)):
    ax.text(x - g, va + 1.5, f"{va}", ha="center", va="bottom", fontsize=10, color="#404040")
    ax.text(x + g, vb + 1.5, f"{vb}", ha="center", va="bottom", fontsize=10, color="#404040")
ax.set_xticks(list(xs)); ax.set_xticklabels([n for n, _ in T], fontsize=13)
ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20)); ax.tick_params(axis="y", labelsize=11, length=0)
ax.grid(axis="y", color="#d9d9d9"); ax.set_axisbelow(True)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=2, frameon=False, fontsize=12, handlelength=1, handleheight=1)
fig.tight_layout(); fig.savefig("lora/figures/type_em_causal_20261108.png")
