"""Molmo2-4B vs Qwen3-VL-4B/2B 비교 표 집계 (검증보고서_Molmo2_vs_Qwen3VL.md용).

모델별: 파라미터 수(safetensors 헤더), 체크포인트·ONNX·엔진 크기, Edge-LLM 프로파일(시각부/prefill/decode GPU 시간
평균·중앙값·p95, 최대 GPU 메모리), 질의당 wall-clock(request_wall_ms 평균·중앙값·p95), 입력·영상 토큰 수,
정확도(EM·lenient, 유형별), Molmo2 Edge-LLM 대비 McNemar(compare_runs.py와 같은 연속성 보정 카이제곱).
결과: edgellm_check/ws_qwen/summary.md, summary.json

usage: python3 -I summarize_molmo2_vs_qwen.py   (표준 라이브러리만 사용)
"""
import collections
import glob
import json
import math
import os
import statistics
import struct

R = "/home/user/ondeviceAI"
RES = f"{R}/molmo2/results"
MODELS = {
    "Molmo2-4B": dict(
        ckpt=f"{R}/molmo2/hf_cache/hub/models--allenai--Molmo2-4B/snapshots/*/model-*.safetensors",
        vision_prefix=("model.vision_backbone.",),
        onnx=[f"{R}/edgellm_check/ws/molmo2_llm/onnx_fp16/llm", f"{R}/edgellm_check/ws/molmo2_visual/molmo2_visual_fp16.onnx"],
        llm_engine=f"{R}/edgellm_check/ws_runner/molmo2_llm_engine/llm.engine",
        vis_engine=f"{R}/edgellm_check/ws/molmo2_visual/visual_fp16.engine",
        profile=f"{R}/edgellm_check/ws_runner/vrs2001/profile_fp16_timed.json",
        hf=f"{RES}/vrsbench_2001_rules_molmo2_hf_causal_bs1.jsonl",
        el=f"{RES}/vrsbench_2001_rules_molmo2_edgellm_fp16.jsonl",
        samples=f"{R}/edgellm_check/ws_runner/vrs2001/samples.jsonl",
    ),
    "Qwen3-VL-4B": dict(
        ckpt=f"{R}/edgellm_check/hf_cache/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/*/model*.safetensors",
        vision_prefix=("model.visual.",),
        onnx=[f"{R}/edgellm_check/ws_qwen/qwen3vl4b/onnx"],
        llm_engine=f"{R}/edgellm_check/ws_qwen/qwen3vl4b/engines/llm/llm.engine",
        vis_engine=f"{R}/edgellm_check/ws_qwen/qwen3vl4b/engines/visual/visual.engine",
        profile=f"{R}/edgellm_check/ws_qwen/qwen3vl4b/profile2001.json",
        hf=f"{RES}/vrsbench_2001_rules_qwen3vl4b_hf.jsonl",
        el=f"{RES}/vrsbench_2001_rules_qwen3vl4b_edgellm_fp16.jsonl",
    ),
    "Qwen3-VL-2B": dict(
        ckpt=f"{R}/edgellm_check/hf_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/*/model*.safetensors",
        vision_prefix=("model.visual.",),
        onnx=[f"{R}/edgellm_check/ws_qwen/qwen3vl2b/onnx"],
        llm_engine=f"{R}/edgellm_check/ws_qwen/qwen3vl2b/engines/llm/llm.engine",
        vis_engine=f"{R}/edgellm_check/ws_qwen/qwen3vl2b/engines/visual/visual.engine",
        profile=f"{R}/edgellm_check/ws_qwen/qwen3vl2b/profile2001.json",
        hf=f"{RES}/vrsbench_2001_rules_qwen3vl2b_hf.jsonl",
        el=f"{RES}/vrsbench_2001_rules_qwen3vl2b_edgellm_fp16.jsonl",
    ),
}
STAGES = {"vision_encoder": "vision", "multimodal_processing": "vision", "llm_prefill": "prefill", "llm_generation": "decode"}


def du(path):
    if os.path.isfile(path):
        return os.path.getsize(os.path.realpath(path))
    return sum(os.path.getsize(os.path.realpath(os.path.join(d, f))) for d, _, fs in os.walk(path) for f in fs)


def params(pattern, vision_prefix):
    total = vision = 0
    for p in glob.glob(pattern):
        with open(p, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for k, v in header.items():
            if k == "__metadata__":
                continue
            n = math.prod(v["shape"])
            total += n
            vision += n if k.startswith(vision_prefix) else 0
    return total, vision


def mcnemar_p(b, c):
    if b + c == 0:
        return 1.0
    return math.erfc(math.sqrt((abs(b - c) - 1) ** 2 / (b + c) / 2))


def load(path):
    return {r["qid"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else None


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main():
    out = {}
    for name, m in MODELS.items():
        if not os.path.exists(m["profile"]):
            continue
        tot, vis = params(m["ckpt"], m["vision_prefix"])
        prof = json.load(open(m["profile"]))
        info = dict(params_total_b=tot / 1e9, params_vision_b=vis / 1e9, params_llm_b=(tot - vis) / 1e9,
                    ckpt_gb=sum(os.path.getsize(os.path.realpath(p)) for p in glob.glob(m["ckpt"])) / 1e9,
                    onnx_gb=sum(du(p) for p in m["onnx"]) / 1e9,
                    llm_engine_gb=du(m["llm_engine"]) / 1e9, vis_engine_gb=du(m["vis_engine"]) / 1e9,
                    peak_gpu_mb=prof["peak_gpu_memory_mb"], prefill_tokens=prof["prefill"]["average_tokens_per_run"],
                    image_tokens=prof["multimodal"]["total_image_tokens"] / max(1, prof["multimodal"]["total_images"]))
        for s in prof["stages"]:
            st = s["gpu_time_stats"]
            info[STAGES[s["stage_id"]]] = dict(mean=st["mean_ms"], median=st["median_ms"], p95=st["p95_ms"],
                                               count=st["count"])
        el, hf = load(m["el"]), load(m["hf"])
        walls = [r["wall_ms"] for r in el.values() if r.get("wall_ms") is not None]
        if walls:
            info["query"] = dict(mean=statistics.mean(walls), median=statistics.median(walls), p95=pct(walls, 0.95))
        info["gen_tokens_mean"] = statistics.mean(r["n_gen"] for r in el.values())
        for tag, runs in (("el", el), ("hf", hf)):
            if runs is None:
                continue
            by = collections.defaultdict(list)
            for r in runs.values():
                by["ALL"].append(r)
                by[r["type"]].append(r)
            info[f"acc_{tag}"] = {t: dict(n=len(rs), em=100 * sum(r["em"] for r in rs) / len(rs),
                                          lenient=100 * sum(r["lenient"] for r in rs) / len(rs)) for t, rs in by.items()}
        if hf is not None:
            info["el_vs_hf_same_text"] = sum(el[q]["pred"] == hf[q]["pred"] for q in el) / len(el)
            b = sum(hf[q]["em"] and not el[q]["em"] for q in el)
            c = sum(not hf[q]["em"] and el[q]["em"] for q in el)
            info["el_vs_hf_mcnemar"] = dict(b=b, c=c, p=mcnemar_p(b, c))
        out[name] = (info, el)

    base = out["Molmo2-4B"][1]
    summary = {}
    for name, (info, el) in out.items():
        if name != "Molmo2-4B":
            info["vs_molmo2"] = {}
            types = sorted({r["type"] for r in base.values()}) + ["ALL"]
            for t in types:
                qs = [q for q in base if t == "ALL" or base[q]["type"] == t]
                b = sum(base[q]["em"] and not el[q]["em"] for q in qs)
                c = sum(not base[q]["em"] and el[q]["em"] for q in qs)
                info["vs_molmo2"][t] = dict(n=len(qs), molmo2=100 * sum(base[q]["em"] for q in qs) / len(qs),
                                            other=100 * sum(el[q]["em"] for q in qs) / len(qs), b=b, c=c,
                                            p=mcnemar_p(b, c))
        summary[name] = info
    json.dump(summary, open(f"{R}/edgellm_check/ws_qwen/summary.json", "w"), indent=1, ensure_ascii=False)

    names = list(summary)
    L = []
    row = lambda label, f: L.append(f"| {label} | " + " | ".join(f(summary[n]) for n in names) + " |")
    L.append("| 항목 | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    row("파라미터 전체 (B)", lambda i: f"{i['params_total_b']:.2f}")
    row("　LLM (B)", lambda i: f"{i['params_llm_b']:.2f}")
    row("　시각부 (B)", lambda i: f"{i['params_vision_b']:.3f}")
    row("영상 토큰 (512×512)", lambda i: f"{i['image_tokens']:.0f}")
    row("입력 토큰 (평균)", lambda i: f"{i['prefill_tokens']:.0f}")
    row("체크포인트 (GB)", lambda i: f"{i['ckpt_gb']:.2f}")
    row("ONNX (GB)", lambda i: f"{i['onnx_gb']:.2f}")
    row("LLM 엔진 (GB)", lambda i: f"{i['llm_engine_gb']:.2f}")
    row("시각 엔진 (GB)", lambda i: f"{i['vis_engine_gb']:.2f}")
    row("최대 GPU 메모리 (GB)", lambda i: f"{i['peak_gpu_mb'] / 1024:.1f}")
    for key, label in (("vision", "시각부 (ms)"), ("prefill", "prefill (ms)"), ("decode", "decode step (ms)"),
                       ("query", "질의당 전체 wall (ms)")):
        row(f"{label} 평균 / 중앙값 / p95",
            lambda i, k=key: f"{i[k]['mean']:.1f} / {i[k]['median']:.1f} / {i[k]['p95']:.1f}" if k in i else "-")
    row("생성 토큰 (평균)", lambda i: f"{i['gen_tokens_mean']:.2f}")
    row("HF EM / lenient", lambda i: f"{i['acc_hf']['ALL']['em']:.1f} / {i['acc_hf']['ALL']['lenient']:.1f}")
    row("Edge-LLM EM / lenient", lambda i: f"{i['acc_el']['ALL']['em']:.1f} / {i['acc_el']['ALL']['lenient']:.1f}")
    row("Edge-LLM vs HF (O→X / X→O, p)", lambda i: "{b} / {c}, {p:.3f}".format(**i["el_vs_hf_mcnemar"]))
    row("Edge-LLM vs HF 답변 동일", lambda i: f"{100 * i['el_vs_hf_same_text']:.1f}%")
    L.append("")
    L.append("| 유형 | n | " + " | ".join(f"{n} EL" for n in names) + " | " + " | ".join(
        f"{n} vs Molmo2 (O→X/X→O, p)" for n in names[1:]) + " |")
    L.append("|---|---|" + "---|" * (2 * len(names) - 1))
    types = sorted(summary["Molmo2-4B"]["acc_el"], key=lambda t: -summary["Molmo2-4B"]["acc_el"][t]["n"])
    for t in types:
        cells = [f"{summary[n]['acc_el'][t]['em']:.1f}" for n in names]
        cmp = ["{b}/{c}, {p:.3f}".format(**summary[n]["vs_molmo2"][t]) for n in names[1:]]
        L.append(f"| {t} | {summary['Molmo2-4B']['acc_el'][t]['n']} | " + " | ".join(cells + cmp) + " |")
    open(f"{R}/edgellm_check/ws_qwen/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
