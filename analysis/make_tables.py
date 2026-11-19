#!/usr/bin/env python3
"""저장소에 포함된 결과 파일만으로 결과 표를 다시 만든다.

출력 (기본 results/tables/, --out 으로 변경)
- results_final.csv : 모델 × 설정별 정확도, 정합성, 지연, 메모리, 전력, 엔진 크기
- type_accuracy.csv : 질문 유형별 EM

실행: python3 analysis/make_tables.py [--out DIR]
"""
import argparse
import csv
import glob
import gzip
import json
import math
import os
import re
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
A100 = os.path.join(RES, "a100")
ORIN = os.path.join(RES, "orin")
CMP = os.path.join(ORIN, "compare")
AN = os.path.join(ORIN, "analysis")
REF = os.path.join(A100, "ref_fp16_v16k_2b")
FP16_ATTEMPT = os.path.join(ORIN, "qwen3vl2b_r16_ep1_fp16_v16k")

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(RES, "tables"))
OUT = ap.parse_args().out

M = [  # (보드 폴더 접두, 결과 파일 이름, 표 이름, A100 지연 키)
    ("molmo2_causal_r16_ep1", "molmo2_merged_causal_r16_ep1", "molmo2_4b", "molmo2"),
    ("qwen3vl4b_r16_ep1", "qwen3vl4b_lora_r16_ep1", "qwen3vl_4b", "qwen3vl4b"),
    ("qwen3vl2b_r16_ep1", "qwen3vl2b_lora_r16_ep1", "qwen3vl_2b", "qwen3vl2b"),
]
COLS = ["model", "setting", "em_test_full", "em_ci_lo", "em_ci_hi", "em_2000_board", "em_2000_a100", "consist_diff", "consist_ci_lo", "consist_ci_hi",
        "resp_agree", "lat_mean_ms", "lat_p95_ms", "lat_rep_sd", "t_visual_ms", "t_prefill_ms", "t_gen_ms", "t_other_ms", "input_tokens", "gen_tokens",
        "ram_idle_mb", "ram_peak_mb", "ram_delta_mb", "rss_max_mb", "power_idle_w", "power_infer_w", "energy_j", "energy_net_j", "engine_llm_gb",
        "engine_vis_gb", "build_llm_s", "build_llm_rss_mb", "build_vis_s", "build_vis_rss_mb", "tj_max_c", "a100_lat_ms", "a100_peak_mib", "note"]
# Qwen3-VL-4B 전체 어휘 보드 측정은 전원 모드 MAXN_SUPER, CPU 클럭 비고정이라 지연·전력·에너지를 비움 (메모리·정합성만 유효)
INVALID_4B_FULL = ["lat_mean_ms", "lat_p95_ms", "lat_rep_sd", "t_visual_ms", "t_prefill_ms", "t_gen_ms", "t_other_ms", "power_idle_w", "power_infer_w",
                   "energy_j", "energy_net_j"]


def jl(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f:
        return [json.loads(l) for l in f]


def em_ci(rows):  # EM(%)과 Wald 95% CI
    n = len(rows); p = sum(bool(r["em"]) for r in rows) / n
    h = 1.96 * math.sqrt(p * (1 - p) / n)
    return 100 * p, 100 * (p - h), 100 * (p + h), n


def em_on(p, qids):
    d = {r["qid"]: r for r in jl(p)}
    return 100 * sum(bool(d[q]["em"]) for q in qids) / len(qids)


def twm(log):  # 빌드 로그의 Total Weights Memory (GB, 10^9 B)
    x = re.findall(r"Total Weights Memory: (\d+) bytes", open(log, errors="replace").read()) if os.path.exists(log) else []
    return int(x[-1]) / 1e9 if x else None


an = {"v16k": json.load(open(f"{AN}/analysis.json")), "full": json.load(open(f"{AN}/analysis_full.json"))}
th = {"v16k": json.load(open(f"{AN}/thermal_clock.json")), "full": json.load(open(f"{AN}/thermal_clock_full.json"))}
lat_a100 = json.load(open(f"{A100}/latency/summary.json"))
SAMPLES = jl(os.path.join(ROOT, "data/splits/consistency2000_requests_samples.jsonl"))
Q2000 = [s["qid"] for s in SAMPLES]

rows = []
for pre, rn, name, lk in M:
    # A100 행
    for setting, suf, lkey in [("fp16_a100", "fp16", f"{lk}_fp16"), ("int4_a100", "int4", f"{lk}_int4"), ("int4_v16k_a100", "int4_v16k", None)]:
        r = dict(model=name, setting=setting)
        e, lo, hi, n = em_ci(jl(f"{A100}/test_full/vrsbench_full_rules_{rn}_edgellm_{suf}.jsonl.gz"))
        r.update(em_test_full=e, em_ci_lo=lo, em_ci_hi=hi, em_2000_a100=em_on(f"{A100}/subset2001/vrsbench_2001_rules_{rn}_edgellm_{suf}.jsonl", Q2000))
        if lkey:
            r.update(a100_lat_ms=lat_a100[lkey]["mean"], a100_peak_mib=lat_a100[lkey]["peak_gpu_mib"])
        r["note"] = "A100 Edge-LLM, 엔진 2048/4096/1"
        rows.append(r)
    # 보드 행
    for s, setting, cmp in [("v16k", "int4_v16k_orin", f"{pre}_int4_v16k_compare.json"), ("full", "int4_full_orin", f"{pre}_int4_full_vs_a100bundlecheck.json")]:
        P = f"{pre}_int4_{s}"; a = an[s][P]; c = json.load(open(f"{CMP}/{cmp}"))["stats"]["ALL"]
        r = dict(model=name, setting=setting, em_2000_board=c["acc_b"], em_2000_a100=c["acc_a"], consist_diff=c["diff_pp"], consist_ci_lo=c["ci_lo"],
                 consist_ci_hi=c["ci_hi"], resp_agree=c["agree"], lat_mean_ms=a["lat_mean_ms"], lat_p95_ms=a["lat_p95_ms"], lat_rep_sd=a["lat_mean_sd"],
                 t_visual_ms=a["stage_ms"]["visual"], t_prefill_ms=a["stage_ms"]["prefill"], t_gen_ms=a["stage_ms"]["generation"], t_other_ms=a["stage_ms"]["other"],
                 input_tokens=a["tokens"]["prefill_per_query"], gen_tokens=a["tokens"]["generated_per_query"], ram_idle_mb=a["ram"]["idle_median_mb"],
                 ram_peak_mb=a["ram"]["peak_mb"], ram_delta_mb=a["ram"]["delta_mb"], rss_max_mb=a["chunk_maxrss_mb"]["max"], power_idle_w=a["idle_W"],
                 power_infer_w=a["infer_W"], energy_j=a["energy_J_per_query"], energy_net_j=a["energy_net_J_per_query"],
                 engine_llm_gb=twm(f"{ORIN}/{P}/build_llm.log"),
                 engine_vis_gb=twm(f"{ORIN}/{pre}_int4_v16k/build_visual.log"),  # 전체 어휘 측정의 시각 엔진은 v16k 시각 엔진과 같은 파일
                 tj_max_c=th[s][P]["tj_max"])
        bl = a["build"].get("build_llm.log", {}); bv = a["build"].get("build_visual.log", {})
        if bl.get("elapsed_s"): r.update(build_llm_s=bl["elapsed_s"][0], build_llm_rss_mb=bl["maxrss_mb"][0])
        if bv.get("elapsed_s"): r.update(build_vis_s=bv["elapsed_s"][0], build_vis_rss_mb=bv["maxrss_mb"][0])
        notes = ["Orin Nano 8GB 25W, CPU 1344 고정"] if not (s == "full" and pre.startswith("qwen3vl4b")) else []
        if s == "v16k": notes.append("A100 기준 = 2,001 v16k(2048/4096)")
        else: notes.append("A100 기준 = 1차 번들 사전 확인(1024/1152/1); 시각 엔진 = v16k 시각 엔진 링크")
        if s == "full" and pre.startswith("molmo2"): notes.append("재부팅 직후 유휴 540MB(v16k보다 743MB 낮음), 언어 엔진 = v1 엔진 링크(빌드 time 없음)")
        if s == "full" and pre.startswith("qwen3vl4b"):
            notes.append("지연·전력·에너지 칸 비움: MAXN_SUPER, CPU 비고정으로 제외 (1344 고정 0%, tj 72.6C). 메모리·정합성만 유효")
            for col in INVALID_4B_FULL: r[col] = None
        r["note"] = "; ".join(notes)
        rows.append(r)

# Qwen3-VL-2B FP16 + v16k 보드 탑재 시도 (빌드 실패)
txt = open(f"{FP16_ATTEMPT}/build_llm.log", errors="replace").read()
el = re.findall(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", txt)[0]
sec = sum(float(v) * 60 ** i for i, v in enumerate(reversed(el.split(":"))))
rss = int(re.findall(r"Maximum resident set size \(kbytes\): (\d+)", txt)[0]) / 1024
wall = [x["request_wall_ms"] for x in json.load(open(f"{REF}/consistency2000_out.json"))["responses"]]
peak = max(json.load(open(p))["peak_gpu_memory_mb"] for p in glob.glob(f"{REF}/cons_c*_profile.json"))
emref = json.load(open(f"{REF}/compare_fp16v16k_vs_int4v16k.json"))["stats"]["ALL"]["acc_a"]
rows.append(dict(model="qwen3vl_2b", setting="fp16_v16k_orin_attempt", em_2000_a100=emref, build_llm_s=sec, build_llm_rss_mb=rss, a100_lat_ms=st.mean(wall),
                 a100_peak_mib=peak, note="보드 llm_build 실패(OutOfMemory, 요청 5,704,487,168 B) → 탑재 불가. A100 FP16 v16k(1024/1152/1) Total Weights Memory 4,999,872,512 B. a100_* = A100 FP16 v16k 기준"))

os.makedirs(OUT, exist_ok=True)
fmt = lambda v: "" if v is None else (f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v))
with open(f"{OUT}/results_final.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(COLS)
    for r in rows: w.writerow([fmt(r.get(c)) for c in COLS])

TYPES = sorted({s["type"] for s in SAMPLES})
with open(f"{OUT}/type_accuracy.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "setting", "subset", "type", "n", "em"])
    for pre, rn, name, lk in M:
        for setting, suf in [("fp16_a100", "fp16"), ("int4_a100", "int4"), ("int4_v16k_a100", "int4_v16k")]:
            X = jl(f"{A100}/test_full/vrsbench_full_rules_{rn}_edgellm_{suf}.jsonl.gz")
            for t in sorted({x["type"] for x in X}):
                xs = [x for x in X if x["type"] == t]; w.writerow([name, setting, "test_full", t, len(xs), round(100 * sum(bool(x["em"]) for x in xs) / len(xs), 3)])
        for setting, fn in [("int4_v16k_orin", f"vrsbench_2000_rules_{rn}_orin_int4_v16k.jsonl"), ("int4_full_orin", f"vrsbench_2000_rules_{rn}_orin_int4_full.jsonl")]:
            X = jl(f"{CMP}/{fn}")
            for t in TYPES:
                xs = [x for x in X if x["type"] == t]; w.writerow([name, setting, "board_2000", t, len(xs), round(100 * sum(bool(x["em"]) for x in xs) / len(xs), 3)])
    X = jl(f"{REF}/vrsbench_2000_rules_qwen3vl2b_lora_r16_ep1_a100_fp16_v16k_orinprofile.jsonl")
    for t in TYPES:
        xs = [x for x in X if x["type"] == t]; w.writerow(["qwen3vl_2b", "fp16_v16k_a100_orinprofile", "a100_2000", t, len(xs), round(100 * sum(bool(x["em"]) for x in xs) / len(xs), 3)])
print(f"{len(rows)} rows -> {OUT}")
