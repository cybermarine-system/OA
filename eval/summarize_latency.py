"""A100 단독 지연 요약 (quant/run_latency_a100.sh 결과): 요청별 wall-clock 평균·중앙값·p95, 단계별 GPU 시간, 최대 GPU 메모리,
그리고 앞선 2,001문항 결과와 출력이 같은지(결정성) 확인.

usage: molmo2/.venv/bin/python quant/scripts/summarize_latency.py → quant/latency_a100/summary.md
"""
import json
import os
import statistics

R = "/home/user/ondeviceAI"
L = f"{R}/quant/latency_a100"
PREV = {
    "molmo2_fp16": f"{R}/edgellm_check/ws_lora/causal_r16_ep1/edgellm_out_fp16.json",
    "molmo2_int4": f"{R}/quant/molmo2_causal_r16_ep1/edgellm_out_int4.json",
    "qwen3vl4b_fp16": f"{R}/quant/qwen3vl4b_r16_ep1/fp16/out2001.json",
    "qwen3vl4b_int4": f"{R}/quant/qwen3vl4b_r16_ep1/int4/out2001.json",
    "qwen3vl2b_fp16": f"{R}/quant/qwen3vl2b_r16_ep1/fp16/out2001.json",
    "qwen3vl2b_int4": f"{R}/quant/qwen3vl2b_r16_ep1/int4/out2001.json",
}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main():
    rows = ["| 엔진 | n | wall 평균 / 중앙값 / p95 (ms) | 시각부 GPU 평균 (ms) | prefill 평균 (ms) | decode step 평균 (ms) | 생성 토큰 평균 | 최대 GPU 메모리 (MiB) | 이전 실행과 출력 일치 |",
            "|---|---|---|---|---|---|---|---|---|"]
    js = {}
    for k, prev in PREV.items():
        p = f"{L}/{k}/out.json"
        if not os.path.exists(p):
            rows.append(f"| {k} | 미측정 | | | | | | | |")
            continue
        out = json.load(open(p))["responses"]
        prof = json.load(open(f"{L}/{k}/profile.json"))
        x = [r["request_wall_ms"] for r in out]
        vis = [s for s in prof.get("stages", []) if s.get("stage_id") in ("vision_encoder", "multimodal_processing")]
        old = json.load(open(prev))["responses"]
        same = sum(a["output_text"] == b["output_text"] for a, b in zip(out, old)) / len(out) * 100
        js[k] = dict(n=len(x), mean=statistics.mean(x), median=statistics.median(x), p95=pct(x, 0.95),
                     vision_ms=vis[0]["average_time_per_run_ms"] if vis else None,
                     prefill_ms=prof["prefill"]["average_time_per_run_ms"],
                     decode_ms=prof["generation"]["average_time_per_token_ms"],
                     gen_tokens=prof["generation"]["average_tokens_per_run"],
                     peak_gpu_mib=prof["peak_gpu_memory_mb"], same_as_prev=same)
        j = js[k]
        rows.append(f"| {k} | {j['n']} | {j['mean']:.1f} / {j['median']:.1f} / {j['p95']:.1f} | "
                    f"{j['vision_ms']:.1f} | {j['prefill_ms']:.1f} | {j['decode_ms']:.2f} | {j['gen_tokens']:.2f} | "
                    f"{j['peak_gpu_mib']:,.0f} | {same:.1f}% |")
    txt = "# A100 단독 지연 (2,001문항, 배치 1, greedy, 최대 48토큰, 다른 GPU 작업 없음)\n\n" + "\n".join(rows) + "\n"
    open(f"{L}/summary.md", "w").write(txt)
    json.dump(js, open(f"{L}/summary.json", "w"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()
