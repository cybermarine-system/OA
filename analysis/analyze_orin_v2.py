#!/usr/bin/env python3
"""Orin v2 측정 결과 분석 (2026-11-17, OA2). 읽기만 하고 결과는 analysis/ 아래에 씀.

입력: OA2/orin_results_v2/<패키지>/{summary.json, marks.txt, tegrastats.log, eff_rep*_c*.log/_profile.json, cons_c*.log, build_*.log}
계산식
- 지연: summary.json의 반복별 mean/p95(분할마다 앞 10개 제외, 반복당 500개) → 5개 반복의 평균 ± 표본표준편차
- 단계 분할: 각 효율 분할 profile.json(110개, 워밍업 포함)
    시각 엔진 = stages[multimodal_processing(Molmo2) 또는 vision_encoder(Qwen)].mean, 입력 처리(prefill) = stages[llm_prefill].mean,
    답 생성 = stages[llm_generation].total_gpu_time_ms / 요청 수(110), 기타(호스트 전처리·토크나이즈 등) = request_wall_ms 평균(110개) − 세 값
    → 25개 분할 평균
- 전력: tegrastats "VDD_IN a/b/cmW"의 a(현재값). 보드 시간대 = KST(서버와 같음, marks와 대조해 확인)
    유휴 = idle_start~idle_end 안의 줄 평균
    추론 구간 = 효율 분할 로그에서 "Progress: 11/110"(11번째 요청 시작 = 워밍업 10개 끝) ~ "Processing complete" (측정 100개만)
    추론 평균 전력 = 추론 구간 25개 안의 줄 평균 (tegrastats 시각은 초 단위 → 줄 시각 t를 [t, t+1) 로 보고 구간에 완전히 포함된 초만 사용)
    반복 구간 평균 전력 = rep{i}_start~rep{i}_end 안의 줄 평균(엔진 적재 포함)
- 에너지/질의 = 추론 평균 전력 × (추론 구간 총 길이) / 질의 수(2,500)  [J]
    순증 에너지/질의 = (추론 평균 전력 − 유휴 전력) × 같은 시간 / 2,500
- RAM: summary.json(idle_median_mb, peak_mb) + tegrastats 첫 줄 RAM
- tegrastats 공백: 연속 두 줄의 시각 차가 2초 이상인 곳
"""
import datetime
import glob
import json
import os
import re
import statistics as st
import sys

BASE = "/home/user/OA2/orin_results_v2"
OUT = os.path.join(BASE, "analysis")
# 사용법: analyze_orin_v2.py [접미사(기본 v16k)] → 모델 폴더 *_int4_<접미사>, 출력 analysis.json(v16k) / analysis_<접미사>.json
SUF = sys.argv[1] if len(sys.argv) > 1 else "v16k"
MODELS = [f"molmo2_causal_r16_ep1_int4_{SUF}", f"qwen3vl4b_r16_ep1_int4_{SUF}", f"qwen3vl2b_r16_ep1_int4_{SUF}"]
OUTNAME = "analysis.json" if SUF == "v16k" else f"analysis_{SUF}.json"
KST = datetime.timezone(datetime.timedelta(hours=9))
TS = re.compile(r"^(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d) RAM (\d+)/(\d+)MB.*?VDD_IN (\d+)mW/(\d+)mW/(\d+)mW VDD_CPU_GPU_CV (\d+)mW.*?VDD_SOC (\d+)mW")
LOGT = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\.(\d{3})\]")


def epoch(s):
    return datetime.datetime.strptime(s, "%m-%d-%Y %H:%M:%S").replace(tzinfo=KST).timestamp()


def read_tegra(p):
    rows = []
    for line in open(p):
        m = TS.match(line)
        if m:
            rows.append(dict(t=epoch(m.group(1)), ram=int(m.group(2)), vin=int(m.group(4)), vcg=int(m.group(7)), vsoc=int(m.group(8))))
    return rows


def window(rows, a, b, key="vin"):
    xs = [r[key] for r in rows if r["t"] >= a and r["t"] + 1 <= b]
    return xs


def log_times(p, day_epoch):
    """측정 구간: "Progress: 11/110"(11번째 요청 시작 = 워밍업 10개 끝) ~ "Processing complete"(마지막 요청 끝), epoch"""
    a = b = None
    for line in open(p, errors="replace"):
        if "Progress: 11/" in line or "Processing complete" in line:
            h, mi, s, ms = map(int, LOGT.match(line).groups())
            t = day_epoch + h * 3600 + mi * 60 + s + ms / 1000
            if "Progress" in line:
                a = t
            else:
                b = t
    return a, b


def time_v(p):
    txt = open(p, errors="replace").read()
    el = re.findall(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", txt)
    rss = re.findall(r"Maximum resident set size \(kbytes\): (\d+)", txt)
    def sec(x):
        parts = [float(v) for v in x.split(":")]
        return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))
    return [sec(x) for x in el], [int(x) for x in rss]


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for M in MODELS:
        d = os.path.join(BASE, M)
        S = json.load(open(os.path.join(d, "summary.json")))
        marks = {l.split()[0]: float(l.split()[1]) for l in open(os.path.join(d, "marks.txt"))}
        rows = read_tegra(os.path.join(d, "tegrastats.log"))
        r = {}
        # 지연
        means = [x["mean_ms"] for x in S["latency_reps"]]
        p95s = [x["p95_ms"] for x in S["latency_reps"]]
        r["lat_mean_ms"], r["lat_mean_sd"] = st.mean(means), st.stdev(means)
        r["lat_p95_ms"], r["lat_p95_sd"] = st.mean(p95s), st.stdev(p95s)
        r["lat_all"] = S["latency_all"]
        # 단계 분할 (profile)
        vis, pre, gen, wall, ptok, gtok, itok = [], [], [], [], [], [], []
        for i in range(1, 6):
            for k in range(1, 6):
                P = json.load(open(os.path.join(d, f"eff_rep{i}_c{k}_profile.json")))
                stg = {s["stage_id"]: s for s in P["stages"]}
                n = P["prefill"]["total_runs"]
                vk = "multimodal_processing" if "multimodal_processing" in stg else "vision_encoder"  # Molmo2 runner / Qwen
                vis.append(stg[vk]["gpu_time_stats"]["mean_ms"])
                pre.append(stg["llm_prefill"]["gpu_time_stats"]["mean_ms"])
                gen.append(stg["llm_generation"]["total_gpu_time_ms"] / n if "llm_generation" in stg else 0.0)
                ptok.append(P["prefill"]["average_tokens_per_run"])
                gtok.append(P["generation"]["generated_tokens"] / n)
                itok.append(P["multimodal"]["total_image_tokens"] / max(1, P["multimodal"]["total_images"]))
                R = json.load(open(os.path.join(d, f"eff_rep{i}_c{k}.json")))["responses"]
                wall.append(st.mean(x["request_wall_ms"] for x in R))
        r["stage_ms"] = dict(visual=st.mean(vis), prefill=st.mean(pre), generation=st.mean(gen),
                             other=st.mean(wall) - st.mean(vis) - st.mean(pre) - st.mean(gen), wall_110=st.mean(wall))
        r["tokens"] = dict(prefill_per_query=st.mean(ptok), generated_per_query=st.mean(gtok), image_tokens_per_image=st.mean(itok))
        # 전력
        day = datetime.datetime.fromtimestamp(marks["idle_start"], KST).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        idle = window(rows, marks["idle_start"], marks["idle_end"])
        r["idle_W"] = st.mean(idle) / 1000
        r["idle_n"] = len(idle)
        inf_v, inf_cg, inf_soc, dur = [], [], [], 0.0
        for i in range(1, 6):
            for k in range(1, 6):
                a, b = log_times(os.path.join(d, f"eff_rep{i}_c{k}.log"), day)
                dur += b - a
                inf_v += window(rows, a, b); inf_cg += window(rows, a, b, "vcg"); inf_soc += window(rows, a, b, "vsoc")
        r["infer_W"] = st.mean(inf_v) / 1000
        r["infer_cpu_gpu_cv_W"] = st.mean(inf_cg) / 1000
        r["infer_soc_W"] = st.mean(inf_soc) / 1000
        r["infer_n"] = len(inf_v)
        r["infer_dur_s"] = dur
        r["sec_per_query_from_log"] = dur / 2500
        r["energy_J_per_query"] = r["infer_W"] * dur / 2500
        r["energy_net_J_per_query"] = (r["infer_W"] - r["idle_W"]) * dur / 2500
        rep_w = [st.mean(window(rows, marks[f"rep{i}_start"], marks[f"rep{i}_end"])) / 1000 for i in range(1, 6)]
        r["rep_window_W"] = rep_w
        r["rep_window_W_mean"] = st.mean(rep_w)
        # RAM
        r["ram"] = dict(first_line_mb=rows[0]["ram"], idle_median_mb=S["ram"]["idle_median_mb"], peak_mb=S["ram"]["peak_mb"],
                        delta_mb=S["ram"]["peak_mb"] - S["ram"]["idle_median_mb"], cons_peak_mb=S["ram"]["cons_peak_mb"])
        # 프로세스 RSS (분할 로그의 /usr/bin/time)
        rss = []
        for p in glob.glob(os.path.join(d, "*_c*.log")):
            rss += time_v(p)[1]
        r["chunk_maxrss_mb"] = dict(max=max(rss) / 1024, min=min(rss) / 1024, n=len(rss))
        # 빌드
        r["build"] = {}
        for b in ["build_llm.log", "build_visual.log"]:
            if not os.path.exists(os.path.join(d, b)):
                continue  # _full은 시각 엔진 재사용이라 build_visual.log 없음
            el, rs = time_v(os.path.join(d, b))
            r["build"][b] = dict(elapsed_s=el, maxrss_mb=[x / 1024 for x in rs])
        # tegrastats 공백
        gaps = []
        for x, y in zip(rows, rows[1:]):
            if y["t"] - x["t"] >= 2:
                gaps.append((datetime.datetime.fromtimestamp(x["t"], KST).strftime("%H:%M:%S"), y["t"] - x["t"]))
        span = rows[-1]["t"] - rows[0]["t"] + 1
        r["tegra"] = dict(lines=len(rows), span_s=span, lines_per_s=len(rows) / span, gaps_ge2s=gaps[:20], n_gaps=len(gaps))
        res[M] = r
    json.dump(res, open(os.path.join(OUT, OUTNAME), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
