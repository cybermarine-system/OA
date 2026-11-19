"""epoch별 학습·검증 손실 그래프, 발표용 (lora/figures/loss_by_epoch_20261108.png)"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import koreanize_matplotlib  # noqa: F401

E = 2655
R = "lora/runs/"
def load(run, off):
    L = [json.loads(l) for l in open(R + run + "/train_log.jsonl")]
    val = [((r["step"] + off) / E, r["val_loss"]) for r in L if "val_loss" in r and r["step"] > 0]
    tr = [r for r in L if "loss" in r and "val_loss" not in r]
    sm = []
    for a in range(0, 6000, 100):  # 100스텝 평균
        seg = [r["loss"] for r in tr if a < r["step"] <= a + 100]
        if seg: sm.append(((a + 100 + off) / E, sum(seg) / len(seg)))
    return val, sm
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
for name, c, a, b, v1end in [("양방향", "#eb6834", "bidir_r16", "bidir_r16_plateau", 0.6836),
                             ("causal", "#2a78d6", "causal_r16", "causal_r16_plateau", 0.6886)]:
    v1, t1 = load(a, 0); v2, t2 = load(b, E)
    v, t = v1 + v2, [p for p in t1 + t2 if p[0] >= 500/E]
    ax.plot([x for x, _ in t], [y for _, y in t], color=c, lw=1, alpha=0.4, label=f"{name} 학습 손실")
    ax.plot([x for x, _ in v], [y for _, y in v], color=c, lw=2.2, marker="o", ms=5.5, mec="white", mew=1.3, label=f"{name} 검증 손실")
    ax.axhline(v1end, color=c, lw=0.9, ls=":")
ax.axvline(1, color="#9a9a9a", lw=1, ls="--")
ax.set_xlim(0, 3.2); ax.set_ylim(0.35, 0.85); ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
ax.set_xlabel("학습량 (epoch)", fontsize=11); ax.set_ylabel("손실", fontsize=11)
ax.grid(axis="y", color="#e6e6e6"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.56), frameon=False, fontsize=10, ncol=2)
fig.tight_layout(); fig.savefig("lora/figures/loss_by_epoch_20261108.png")
