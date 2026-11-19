"""Orin 측정 결과 요약 v2 (measure_orin_v2.sh 결과용, 요청 100개 분할): E2E 지연(평균·p95, 반복별·전체), RAM 절대 피크·유휴 대비 증가 (tegrastats).

- 지연: llm_inference 출력의 request_wall_ms (요청 1개 처리 wall-clock, 전처리·시각 엔진·prefill·생성 포함).
  각 반복 = 분할 5개, 분할마다 앞 10개(워밍업) 제외 → 반복당 500개
- RAM: tegrastats "RAM a/bMB". 시각이 찍힌 형식(MM-DD-YYYY HH:MM:SS)이면 marks.txt 구간별로 계산,
  아니면 로그 전체 최대값만 표시. 유휴 기준 = idle 구간 중앙값
- 정합성 2,000 결과(consistency2000_out.json)는 서버로 가져와 채점: quant/scripts 참고 (README)

usage: python3 summarize_orin_v2.py <결과 폴더>
"""
import datetime
import json
import os
import re
import statistics
import sys


def p95(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


def main():
    out = sys.argv[1]
    marks = {}
    for line in open(os.path.join(out, "marks.txt")):
        k, v = line.split()
        marks[k] = float(v)
    allx, rows = [], []
    for i in range(1, 6):
        xs = []
        for k in range(1, 6):  # v2: 분할 5개, 각 분할 앞 10개(워밍업) 제외
            p = os.path.join(out, f"eff_rep{i}_c{k}.json")
            if os.path.exists(p):
                xs += [x["request_wall_ms"] for x in json.load(open(p))["responses"][10:]]
        if not xs:
            continue
        allx += xs
        rows.append((i, len(xs), statistics.mean(xs), p95(xs)))
    ram = []
    pat = re.compile(r"^(\d\d-\d\d-\d{4} \d\d:\d\d:\d\d).*?RAM (\d+)/(\d+)MB")
    pat2 = re.compile(r"RAM (\d+)/(\d+)MB")
    for line in open(os.path.join(out, "tegrastats.log")):
        m = pat.search(line)
        if m:
            t = datetime.datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S").timestamp()
            ram.append((t, int(m.group(2)), int(m.group(3))))
        else:
            m = pat2.search(line)
            if m:
                ram.append((None, int(m.group(1)), int(m.group(2))))
    res = dict(latency_reps=[dict(rep=i, n=n, mean_ms=m, p95_ms=q) for i, n, m, q in rows])
    if allx:
        res["latency_all"] = dict(n=len(allx), mean_ms=statistics.mean(allx), p95_ms=p95(allx))
    total = ram[0][2] if ram else None
    if ram and ram[0][0] is not None:
        idle = [r for t, r, _ in ram if marks["idle_start"] <= t <= marks["idle_end"]]
        base = statistics.median(idle) if idle else None
        per = []
        for i in range(1, 6):
            a, b = marks.get(f"rep{i}_start"), marks.get(f"rep{i}_end")
            if a is None:
                continue
            seg = [r for t, r, _ in ram if a <= t <= b + 1]
            if seg:
                per.append(dict(rep=i, peak_mb=max(seg), delta_mb=max(seg) - base if base else None))
        a, b = marks.get("cons_start"), marks.get("cons_end")
        cons = [r for t, r, _ in ram if a and b and a <= t <= b + 1]
        res["ram"] = dict(total_mb=total, idle_median_mb=base, reps=per,
                          cons_peak_mb=max(cons) if cons else None,
                          peak_mb=max(p["peak_mb"] for p in per) if per else None)
    elif ram:
        res["ram"] = dict(total_mb=total, peak_mb_whole_log=max(r for _, r, _ in ram), note="tegrastats 시각 없음: 구간 구분 불가")
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
