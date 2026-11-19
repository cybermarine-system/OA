"""세 모델(Molmo2-4B ⑤, Qwen3-VL-4B, Qwen3-VL-2B) × PyTorch 병합본·Edge-LLM FP16·INT4 결과 집계 (있는 파일만).

- 2,001문항: PyTorch 병합본 / FP16 / INT4 EM, INT4 − FP16 (95% CI, McNemar, 응답 일치), FP16 − PyTorch
- test 전체 37,409문항: FP16 / INT4 EM, INT4 − FP16 전체·유형별(Holm), 2,001 표본을 뺀 35,408문항 값 병기
- 모델 간: 같은 정밀도끼리 쌍별 차이·McNemar, 유형별 Holm 보정
- 출력: quant/summary_three_models.md, quant/summary_three_models.json

usage: molmo2/.venv/bin/python quant/scripts/summarize_three_models.py
"""
import json
import os
import sys

R = "/home/user/ondeviceAI"
sys.path.insert(0, os.path.join(R, "quant/scripts"))
from compare_pair import holm, load, stats  # noqa: E402

RES = os.path.join(R, "molmo2/results")
MODELS = {
    "Molmo2-4B": dict(hf="vrsbench_2001_rules_molmo2_merged_causal_r16_ep1_causal.jsonl",
                      tag="molmo2_merged_causal_r16_ep1"),
    "Qwen3-VL-4B": dict(hf="vrsbench_2001_rules_qwen3vl4b_lora_r16_ep1_merged_hf.jsonl", tag="qwen3vl4b_lora_r16_ep1"),
    "Qwen3-VL-2B": dict(hf="vrsbench_2001_rules_qwen3vl2b_lora_r16_ep1_merged_hf.jsonl", tag="qwen3vl2b_lora_r16_ep1"),
}
TYPES = ["presence", "quantity", "position", "category", "scene", "color", "shape", "size", "reasoning", "direction"]


def get(name):
    p = os.path.join(RES, name)
    return load(p) if os.path.exists(p) else None


def pair(A, B, keys=None):
    keys = [k for k in (keys or A) if k in A and k in B]
    rows = {"ALL": []}
    for k in keys:
        x, y = A[k], B[k]
        t = (bool(x["em"]), bool(y["em"]), x["pred"].strip() == y["pred"].strip())
        rows["ALL"].append(t)
        rows.setdefault(x["type"], []).append(t)
    res = {t: stats(v) for t, v in rows.items() if v}
    ts = [t for t in res if t != "ALL"]
    for t, p in zip(ts, holm([res[t]["p"] for t in ts])):
        res[t]["p_holm"] = p
    return res


def acc(A, keys=None):
    ks = [k for k in (keys or A) if k in A]
    return 100 * sum(bool(A[k]["em"]) for k in ks) / len(ks) if ks else None


def f(x, d=1):
    return "미측정" if x is None else f"{x:.{d}f}"


def fd(s):
    return f"{s['diff_pp']:+.2f} [{s['ci_lo']:+.2f}, {s['ci_hi']:+.2f}]"


def main():
    md, js = [], {}
    s2001 = None
    data = {}
    for m, c in MODELS.items():
        d = dict(hf=get(c["hf"]),
                 fp16=get(f"vrsbench_2001_rules_{c['tag']}_edgellm_fp16.jsonl"),
                 int4=get(f"vrsbench_2001_rules_{c['tag']}_edgellm_int4.jsonl"),
                 full_fp16=get(f"vrsbench_full_rules_{c['tag']}_edgellm_fp16.jsonl"),
                 full_int4=get(f"vrsbench_full_rules_{c['tag']}_edgellm_int4.jsonl"))
        data[m] = d
        for k in ("fp16", "int4", "hf"):
            if d[k] and s2001 is None:
                s2001 = set(d[k])
    md.append("# 세 모델 LoRA·INT4 집계 (자동 생성)\n")
    md.append("## 1. 2,001문항 (test 층화 표본, seed 0)\n")
    md.append("| 모델 | PyTorch 병합본 EM | Edge-LLM FP16 EM | Edge-LLM INT4 EM | INT4 − FP16 %p [95% CI] | b/c, McNemar p | 응답 일치 | FP16 − PyTorch %p [95% CI] |")
    md.append("|---|---|---|---|---|---|---|---|")
    for m, d in data.items():
        r = dict(hf=acc(d["hf"]) if d["hf"] else None, fp16=acc(d["fp16"]) if d["fp16"] else None,
                 int4=acc(d["int4"]) if d["int4"] else None)
        q = pair(d["fp16"], d["int4"]) if d["fp16"] and d["int4"] else None
        h = pair(d["hf"], d["fp16"]) if d["hf"] and d["fp16"] else None
        js.setdefault(m, {})["s2001"] = dict(acc=r, int4_vs_fp16=q, fp16_vs_hf=h)
        md.append(f"| {m} | {f(r['hf'])} | {f(r['fp16'])} | {f(r['int4'])} | {fd(q['ALL']) if q else '미측정'} | "
                  f"{(str(q['ALL']['b']) + '/' + str(q['ALL']['c']) + ', ' + format(q['ALL']['p'], '.3f')) if q else '미측정'} | "
                  f"{f(q['ALL']['agree']) if q else '미측정'} | {fd(h['ALL']) if h else '미측정'} |")
    md.append("\n### 유형별 INT4 − FP16 (2,001, %p, 괄호는 Holm 보정 p)\n")
    md.append("| 유형 | n | " + " | ".join(MODELS) + " |")
    md.append("|---|---|" + "---|" * len(MODELS))
    for t in TYPES:
        cells, n = [], None
        for m in MODELS:
            q = js[m]["s2001"]["int4_vs_fp16"]
            if q and t in q:
                n = q[t]["n"]
                cells.append(f"{q[t]['diff_pp']:+.2f} [{q[t]['ci_lo']:+.1f}, {q[t]['ci_hi']:+.1f}] ({q[t]['p_holm']:.3f})")
            else:
                cells.append("미측정")
        md.append(f"| {t} | {n or ''} | " + " | ".join(cells) + " |")

    md.append("\n## 2. test 전체 37,409문항 (Edge-LLM, A100)\n")
    md.append("| 모델 | FP16 EM | INT4 EM | INT4 − FP16 %p [95% CI] | b/c, McNemar p | 응답 일치 | 35,408문항(2,001 제외) FP16 / INT4 |")
    md.append("|---|---|---|---|---|---|---|")
    for m, d in data.items():
        F, I = d["full_fp16"], d["full_int4"]
        q = pair(F, I) if F and I else None
        rest = [k for k in (F or I or {}) if s2001 and k not in s2001]
        js[m]["full"] = dict(fp16=acc(F) if F else None, int4=acc(I) if I else None, int4_vs_fp16=q,
                             rest_fp16=acc(F, rest) if F else None, rest_int4=acc(I, rest) if I else None,
                             n_fp16=len(F) if F else 0, n_int4=len(I) if I else 0, n_rest=len(rest))
        j = js[m]["full"]
        md.append(f"| {m} | {f(j['fp16'], 2)} | {f(j['int4'], 2)} | {fd(q['ALL']) if q else '미측정'} | "
                  f"{(str(q['ALL']['b']) + '/' + str(q['ALL']['c']) + ', ' + format(q['ALL']['p'], '.3g')) if q else '미측정'} | "
                  f"{f(q['ALL']['agree']) if q else '미측정'} | {f(j['rest_fp16'], 2)} / {f(j['rest_int4'], 2)} |")
    md.append("\n### 유형별 INT4 − FP16 (test 전체, %p [95% CI], Holm p)\n")
    md.append("| 유형 | n | " + " | ".join(MODELS) + " |")
    md.append("|---|---|" + "---|" * len(MODELS))
    for t in TYPES:
        cells, n = [], None
        for m in MODELS:
            q = js[m]["full"]["int4_vs_fp16"]
            if q and t in q:
                n = q[t]["n"]
                cells.append(f"{q[t]['diff_pp']:+.2f} [{q[t]['ci_lo']:+.2f}, {q[t]['ci_hi']:+.2f}] ({q[t]['p_holm']:.3g})")
            else:
                cells.append("미측정")
        md.append(f"| {t} | {n or ''} | " + " | ".join(cells) + " |")

    md.append("\n## 3. 모델 간 비교 (test 전체, 같은 정밀도, B − A)\n")
    names = list(MODELS)
    js["between"] = {}
    for prec in ("int4", "fp16"):
        md.append(f"\n### {prec.upper()}\n")
        md.append("| A | B | A EM | B EM | B − A %p [95% CI] | b/c, McNemar p | 유형별 Holm p < 0.05 |")
        md.append("|---|---|---|---|---|---|---|")
        for i in range(len(names)):
            for k in range(i + 1, len(names)):
                A, B = data[names[i]]["full_" + prec], data[names[k]]["full_" + prec]
                if not (A and B):
                    md.append(f"| {names[i]} | {names[k]} | 미측정 | 미측정 | 미측정 | 미측정 | |")
                    continue
                q = pair(A, B)
                js["between"][f"{prec}:{names[i]}|{names[k]}"] = q
                sig = ", ".join(f"{t} {q[t]['diff_pp']:+.1f}" for t in TYPES if t in q and q[t]["p_holm"] < 0.05)
                md.append(f"| {names[i]} | {names[k]} | {q['ALL']['acc_a']:.2f} | {q['ALL']['acc_b']:.2f} | {fd(q['ALL'])} | "
                          f"{q['ALL']['b']}/{q['ALL']['c']}, {q['ALL']['p']:.3g} | {sig} |")
        md.append("\n유형별 B − A (%p, Holm p)\n")
        md.append("| 유형 | " + " | ".join(f"{names[i]} → {names[k]}" for i in range(len(names)) for k in range(i + 1, len(names))) + " |")
        md.append("|---|" + "---|" * 3)
        for t in TYPES:
            cells = []
            for i in range(len(names)):
                for k in range(i + 1, len(names)):
                    q = js["between"].get(f"{prec}:{names[i]}|{names[k]}")
                    cells.append(f"{q[t]['diff_pp']:+.1f} ({q[t]['p_holm']:.3g})" if q and t in q else "미측정")
            md.append(f"| {t} | " + " | ".join(cells) + " |")
    open(os.path.join(R, "quant/summary_three_models.md"), "w").write("\n".join(md) + "\n")
    json.dump(js, open(os.path.join(R, "quant/summary_three_models.json"), "w"), indent=1, ensure_ascii=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
