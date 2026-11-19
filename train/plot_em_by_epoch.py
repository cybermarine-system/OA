"""epoch별 VRSBench 2,001문항 EM 그래프, 발표용 (lora/figures/em_by_epoch_20261108.png)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import koreanize_matplotlib  # noqa: F401

E = 2655
xs = [500/E, 1000/E, 1500/E, 2000/E, 2500/E, 1, 2, 3]
S = [("양방향", "#eb6834", [66.0, 68.4, 69.9, 71.5, 71.5, 71.7, 71.6, 72.1]),
     ("causal", "#2a78d6", [65.5, 67.4, 68.9, 70.9, 71.0, 70.8, 70.9, 71.6])]
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
for name, c, ys in S:
    ax.plot(xs, ys, color=c, lw=2.2, marker="o", ms=6, mec="white", mew=1.5, label=name)
    ax.annotate(f"{ys[-1]}", (3, ys[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=11)
ax.axvline(1, color="#9a9a9a", lw=1, ls="--")
ax.set_xlim(0, 3.25); ax.set_ylim(64, 74); ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
ax.set_xlabel("학습량 (epoch)", fontsize=11); ax.set_ylabel("정답률 EM (%)", fontsize=11)
ax.grid(axis="y", color="#e6e6e6"); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=11)
fig.tight_layout(); fig.savefig("lora/figures/em_by_epoch_20261108.png")
