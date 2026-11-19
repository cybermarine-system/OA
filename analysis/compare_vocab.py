#!/usr/bin/env python3
"""Orin 어휘 축소 전(INT4 전체 어휘, full) 대 후(v16k) 비교. 저장소의 results/ 만 읽고 results/tables/compare_vocab.json 에 씀.

입력
- results/orin/analysis/analysis{,_full}.json, thermal_clock{,_full}.json (원시 로그에서 analyze_orin_v2.py, thermal_clock_check.py 로 계산)
- results/orin/<모델>_int4_{v16k,full}/build_llm.log 의 "Total Weights Memory" (언어 엔진 가중치, 입력 임베딩 별도)
- results/orin/compare/*.json, vrsbench_2000_rules_*_orin_int4_{full,v16k}.jsonl
차이 = v16k − full, 비율 = v16k / full
실행: python3 analysis/compare_vocab.py [--out DIR]
"""
import json
import os
import re
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "results", "orin")
A = os.path.join(BASE, "analysis")
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables")); OUT = ap.parse_args().out
MODELS = [("molmo2_causal_r16_ep1", "molmo2_merged_causal_r16_ep1", "Molmo2-4B"),
          ("qwen3vl4b_r16_ep1", "qwen3vl4b_lora_r16_ep1", "Qwen3-VL-4B"),
          ("qwen3vl2b_r16_ep1", "qwen3vl2b_lora_r16_ep1", "Qwen3-VL-2B")]
an = {"v16k": json.load(open(os.path.join(A, "analysis.json"))), "full": json.load(open(os.path.join(A, "analysis_full.json")))}
th = {"v16k": json.load(open(os.path.join(A, "thermal_clock.json"))), "full": json.load(open(os.path.join(A, "thermal_clock_full.json")))}


def twm(m, s):
    t = open(os.path.join(BASE, f"{m}_int4_{s}", "build_llm.log"), errors="replace").read()
    x = re.findall(r"Total Weights Memory: (\d+) bytes", t)
    return int(x[-1]) / 1e9 if x else None


def items(r, m, s):
    return {
        "lat_mean_ms": r["lat_mean_ms"], "lat_mean_sd_ms": r["lat_mean_sd"], "lat_p95_ms": r["lat_p95_ms"], "lat_p95_sd_ms": r["lat_p95_sd"],
        "stage_visual_ms": r["stage_ms"]["visual"], "stage_prefill_ms": r["stage_ms"]["prefill"],
        "stage_generation_ms": r["stage_ms"]["generation"], "stage_other_ms": r["stage_ms"]["other"],
        "gen_tokens_per_query": r["tokens"]["generated_per_query"],
        "proc_maxrss_mb": r["chunk_maxrss_mb"]["max"],
        "ram_idle_median_mb": r["ram"]["idle_median_mb"], "ram_peak_mb": r["ram"]["peak_mb"], "ram_delta_mb": r["ram"]["delta_mb"],
        "infer_W": r["infer_W"], "idle_W": r["idle_W"], "energy_J": r["energy_J_per_query"], "energy_net_J": r["energy_net_J_per_query"],
        "llm_engine_weights_GB": twm(m, s),
        "cpu_all1344_pct": th[s][f"{m}_int4_{s}"]["cpu_all1344_pct"], "tj_max_C": th[s][f"{m}_int4_{s}"]["tj_max"],
    }


def exact_agree(n):
    a = {x["qid"]: x["pred"] for x in map(json.loads, open(os.path.join(BASE, "compare", f"vrsbench_2000_rules_{n}_orin_int4_full.jsonl")))}
    b = {x["qid"]: x["pred"] for x in map(json.loads, open(os.path.join(BASE, "compare", f"vrsbench_2000_rules_{n}_orin_int4_v16k.jsonl")))}
    return sum(a[q] == b[q] for q in a), len(a)


res = {}
for m, n, label in MODELS:
    f, v = items(an["full"][f"{m}_int4_full"], m, "full"), items(an["v16k"][f"{m}_int4_v16k"], m, "v16k")
    if m.startswith("qwen3vl4b"):  # 4B full 지연·전력·에너지 제외 (MAXN_SUPER, CPU 비고정)
        for k in ["lat_mean_ms", "lat_mean_sd_ms", "lat_p95_ms", "lat_p95_sd_ms", "stage_visual_ms", "stage_prefill_ms", "stage_generation_ms", "stage_other_ms",
                  "infer_W", "idle_W", "energy_J", "energy_net_J", "cpu_all1344_pct"]:
            f[k] = None
    rows = {k: dict(full=f[k], v16k=v[k], diff=(None if f[k] is None or v[k] is None else v[k] - f[k]),
                    ratio=(None if not f[k] or v[k] is None else v[k] / f[k])) for k in f}
    cons = {}
    for key, fn in [("a100_bundlecheck_vs_orin_full", f"{m}_int4_full_vs_a100bundlecheck.json"),  # 주 기준: 1차 번들 A100 사전 확인(1024/1152/1)
                    ("a100_2001_vs_orin_full", f"{m}_int4_full_compare.json"), ("a100_vs_orin_v16k", f"{m}_int4_v16k_compare.json"),
                    ("orin_full_vs_v16k", f"{m}_orin_full_vs_v16k.json")]:
        cons[key] = json.load(open(os.path.join(BASE, "compare", fn)))["stats"]["ALL"]
    ea, en = exact_agree(n)
    cons["orin_full_vs_v16k"]["exact_same_pred"] = ea
    # 지연 분해: 지연 평균 차이(v16k−full) 중 각 단계 몫
    dl = rows["lat_mean_ms"]["diff"]
    if dl is None:
        dec = {"excluded": "MAXN_SUPER, CPU 비고정으로 제외"}
    else:
        dec = {s: dict(diff_ms=rows[f"stage_{s}_ms"]["diff"], share_pct=100 * rows[f"stage_{s}_ms"]["diff"] / dl)
               for s in ["visual", "prefill", "generation", "other"]}
        g = rows["gen_tokens_per_query"]["full"]
        # 생성 단계: 질의당 생성 시간 차 ÷ (질의당 생성 토큰) = 토큰당 절감 (decode 스텝 수 = 생성 토큰 − 1 이 아닐 수 있어 둘 다 표시)
        dec["generation"]["per_gen_token_ms"] = -rows["stage_generation_ms"]["diff"] / g
        dec["generation"]["per_decode_step_ms_if_tokens_minus_1"] = -rows["stage_generation_ms"]["diff"] / (g - 1)
    dec["lmhead_removed_GB"] = rows["llm_engine_weights_GB"]["full"] - rows["llm_engine_weights_GB"]["v16k"] if rows["llm_engine_weights_GB"]["full"] else None
    res[m] = dict(label=label, rows=rows, consistency=cons, latency_decomposition=dec)

# 4B 지연 효과 추정: 같은 언어부(Qwen3-4B)를 쓰는 Molmo2의 생성 + prefill 절감 (추정)
mo = res["molmo2_causal_r16_ep1"]["latency_decomposition"]
res["qwen3vl4b_r16_ep1"]["latency_effect_estimate_ms"] = dict(
    value=mo["generation"]["diff_ms"] + mo["prefill"]["diff_ms"], basis="추정: Molmo2 v16k−full 의 생성 + prefill 차이 (같은 Qwen3-4B 언어부, 엔진 가중치 동일)")
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, "compare_vocab.json"), "w"), indent=1, ensure_ascii=False)

# 화면 출력용 표
L = []
fmt = lambda x, d=1: "-" if x is None else f"{x:,.{d}f}"
spec = [("지연 평균 (ms)", "lat_mean_ms", 1), ("지연 평균 반복 SD (ms)", "lat_mean_sd_ms", 2), ("지연 p95 (ms)", "lat_p95_ms", 1), ("지연 p95 반복 SD (ms)", "lat_p95_sd_ms", 2),
        ("├ 시각 (ms)", "stage_visual_ms", 1), ("├ prefill (ms)", "stage_prefill_ms", 1), ("├ 생성 (ms)", "stage_generation_ms", 1), ("└ 기타 (ms)", "stage_other_ms", 1),
        ("프로세스 최대 RSS (MB) [대표]", "proc_maxrss_mb", 0), ("(참고) RAM 유휴 중앙값 (MB)", "ram_idle_median_mb", 0), ("(참고) RAM 피크 (MB)", "ram_peak_mb", 0),
        ("(참고) RAM 증가분 (MB)", "ram_delta_mb", 0), ("추론 구간 평균 전력 (W)", "infer_W", 2), ("질의당 에너지 총 (J)", "energy_J", 2), ("질의당 에너지 순증 (J)", "energy_net_J", 2),
        ("언어 엔진 가중치 (GB, Total Weights Memory, 임베딩 별도)", "llm_engine_weights_GB", 3)]
for m, _, label in MODELS:
    R = res[m]["rows"]
    L.append(f"### {label}" + (" (full의 지연·전력·에너지: MAXN_SUPER, CPU 비고정으로 제외)" if m.startswith("qwen3vl4b") else "") + "\n| 항목 | full | v16k | 차이 (v16k−full) | 비율 (v16k/full) |\n|---|---|---|---|---|")
    for name, k, d in spec:
        r = R[k]
        diff = "-" if r["diff"] is None else format(r["diff"], f"+,.{d}f")
        ratio = "-" if r["ratio"] is None else format(r["ratio"], ".3f")
        L.append(f"| {name} | {fmt(r['full'], d)} | {fmt(r['v16k'], d)} | {diff} | {ratio} |")
    L.append("")
print("\n".join(L))
for m, _, label in MODELS:
    print(label, json.dumps(res[m]["consistency"], ensure_ascii=False))
    print(label, json.dumps(res[m]["latency_decomposition"], ensure_ascii=False))
