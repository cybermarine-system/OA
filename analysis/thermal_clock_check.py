#!/usr/bin/env python3
"""tegrastats 클럭·온도 확인 (2026-11-17, OA2). 읽기만 함.
- 측정 구간 = marks idle_start ~ cons_end (줄 시각 t, [t, t+1)가 구간과 겹치는 줄 모두)
- (1) CPU 6코어 주파수(CPU [x%@f,...])가 모두 1344인 줄의 비율, 아닌 줄의 시각·값
- (2) tj 온도(tj@현재C/최대C의 현재값) 최대, 추론 구간(효율 분할 Progress 11/110 ~ Processing complete) 평균
- GR3D_FREQ는 이 보드 tegrastats에서 사용률(%)만 나옴(주파수 없음) → GPU 클럭 저하는 직접 볼 수 없음.
  간접 확인: 추론 구간 GR3D 사용률 평균의 반복별 추이, 분할별 지연(request_wall_ms 평균)의 처음↔끝 변화
"""
import datetime, json, os, re, statistics as st, sys

BASE = "/home/user/OA2/orin_results_v2"
SUF = sys.argv[1] if len(sys.argv) > 1 else "v16k"  # 사용법: thermal_clock_check.py [v16k|full]
MODELS = [f"molmo2_causal_r16_ep1_int4_{SUF}", f"qwen3vl4b_r16_ep1_int4_{SUF}", f"qwen3vl2b_r16_ep1_int4_{SUF}"]
OUTNAME = "thermal_clock.json" if SUF == "v16k" else f"thermal_clock_{SUF}.json"
KST = datetime.timezone(datetime.timedelta(hours=9))
LT = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\.(\d{3})\]")


def ep(s):
    return datetime.datetime.strptime(s, "%m-%d-%Y %H:%M:%S").replace(tzinfo=KST).timestamp()


def windows(d, day):
    out = []
    for i in range(1, 6):
        for k in range(1, 6):
            a = b = None
            for line in open(os.path.join(d, f"eff_rep{i}_c{k}.log"), errors="replace"):
                if "Progress: 11/" in line or "Processing complete" in line:
                    h, m, s, ms = map(int, LT.match(line).groups())
                    t = day + h * 3600 + m * 60 + s + ms / 1000
                    if "Progress" in line: a = t
                    else: b = t
            out.append((i, k, a, b))
    return out


res = {}
for M in MODELS:
    d = os.path.join(BASE, M)
    marks = {l.split()[0]: float(l.split()[1]) for l in open(os.path.join(d, "marks.txt"))}
    A, B = marks["idle_start"], marks["cons_end"]
    day = datetime.datetime.fromtimestamp(A, KST).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    W = windows(d, day)
    rows = []
    for line in open(os.path.join(d, "tegrastats.log")):
        t = ep(line[:19])
        if t + 1 <= A or t > B:
            continue
        freqs = re.search(r"CPU \[([^\]]*)\]", line).group(1).split(",")
        f = [x.split("@")[1] if "@" in x else x for x in freqs]
        tj = float(re.search(r"tj@([\d.]+)C", line).group(1))
        g = int(re.search(r"GR3D_FREQ (\d+)%", line).group(1))
        rows.append((t, f, tj, g, line[:19]))
    not1344 = [(r[4], r[1]) for r in rows if any(x != "1344" for x in r[1])]
    inf = lambda r: any(a <= r[0] and r[0] + 1 <= b for _, _, a, b in W)
    tj_inf = [r[2] for r in rows if inf(r)]
    tjmax = max(rows, key=lambda r: r[2])
    # 반복별 추론 구간 GR3D, tj
    per_rep = []
    for i in range(1, 6):
        ws = [(a, b) for j, _, a, b in W if j == i]
        sel = [r for r in rows if any(a <= r[0] and r[0] + 1 <= b for a, b in ws)]
        per_rep.append(dict(rep=i, gr3d_mean=st.mean(r[3] for r in sel), tj_mean=st.mean(r[2] for r in sel), tj_max=max(r[2] for r in sel)))
    # 분할별 지연 (첫 분할 vs 마지막 분할)
    lat = []
    for i in range(1, 6):
        for k in range(1, 6):
            R = json.load(open(os.path.join(d, f"eff_rep{i}_c{k}.json")))["responses"][10:]
            lat.append(st.mean(x["request_wall_ms"] for x in R))
    res[M] = dict(lines=len(rows), cpu_all1344=len(rows) - len(not1344), cpu_all1344_pct=100 * (len(rows) - len(not1344)) / len(rows),
                  not1344=not1344[:30], n_not1344=len(not1344),
                  tj_max=tjmax[2], tj_max_at=tjmax[4], tj_infer_mean=st.mean(tj_inf), tj_infer_n=len(tj_inf),
                  per_rep=per_rep, chunk_lat_first_ms=lat[0], chunk_lat_last_ms=lat[-1],
                  chunk_lat_min_ms=min(lat), chunk_lat_max_ms=max(lat))
json.dump(res, open(os.path.join(BASE, "analysis", OUTNAME), "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
