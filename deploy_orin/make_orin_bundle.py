"""Orin 전달 묶음 폴더 생성: 파일은 심볼릭 링크로 두고(복사 안 함), 실제 복사할 목록·용량을 copy_list.txt / bundle_info.json에 기록.

구성: <묶음>/llm/ (언어 INT4 ONNX + 토크나이저·설정), <묶음>/visual/ (시각부 FP16 ONNX + runner 설정), ../common/ (요청·이미지·측정 스크립트·Edge-LLM 패치)
README.md는 별도로 작성 (이 스크립트는 덮어쓰지 않음)

usage: molmo2/.venv/bin/python deploy/make_orin_bundle.py molmo2 | qwen3vl4b | qwen3vl2b | molmo2_v16k | qwen3vl4b_v16k | qwen3vl2b_v16k
"""
import json
import os
import sys

R = "/home/user/ondeviceAI"
D = os.path.join(R, "deploy/orin")

SPECS = {
    "molmo2": dict(
        name="molmo2_causal_r16_ep1_int4",
        llm=os.path.join(R, "quant/molmo2_causal_r16_ep1/onnx_int4/llm"),
        visual_files={
            "molmo2_visual_fp16.onnx": os.path.join(R, "edgellm_check/ws/molmo2_visual/molmo2_visual_fp16.onnx"),
            "new_embedding_fp16.safetensors": os.path.join(R, "edgellm_check/ws_runner/molmo2_visual_engine/new_embedding_fp16.safetensors"),
            "config.json": os.path.join(R, "edgellm_check/ws_runner/molmo2_visual_engine_cpp/config.json"),
            "ref_io.npz": os.path.join(R, "edgellm_check/ws/molmo2_visual/ref_io.npz"),
            "build_check_visual_trt.py": os.path.join(R, "edgellm_check/scripts/build_check_visual_trt.py"),
        }),
    "qwen3vl4b": dict(name="qwen3vl4b_r16_ep1_int4", llm=os.path.join(R, "quant/qwen3vl4b_r16_ep1/int4/onnx/llm"),
                      visual_dir=os.path.join(R, "quant/qwen3vl4b_r16_ep1/int4/onnx/visual")),
    "qwen3vl2b": dict(name="qwen3vl2b_r16_ep1_int4", llm=os.path.join(R, "quant/qwen3vl2b_r16_ep1/int4/onnx/llm"),
                      visual_dir=os.path.join(R, "quant/qwen3vl2b_r16_ep1/int4/onnx/visual")),
    # v2 (2026-11-16): INT4 + 출력 어휘 16,384 축소 (quant/vocab/molmo2_v16k, 세 모델 같은 파일). 시각부는 v1과 같음
    "molmo2_v16k": dict(name="molmo2_causal_r16_ep1_int4_v16k", llm=os.path.join(R, "quant/molmo2_causal_r16_ep1_v16k/onnx_int4/llm"),
                        visual_files="molmo2"),
    "qwen3vl4b_v16k": dict(name="qwen3vl4b_r16_ep1_int4_v16k", llm=os.path.join(R, "quant/qwen3vl4b_r16_ep1/int4_v16k/onnx/llm"),
                           visual_dir=os.path.join(R, "quant/qwen3vl4b_r16_ep1/int4/onnx/visual")),
    "qwen3vl2b_v16k": dict(name="qwen3vl2b_r16_ep1_int4_v16k", llm=os.path.join(R, "quant/qwen3vl2b_r16_ep1/int4_v16k/onnx/llm"),
                           visual_dir=os.path.join(R, "quant/qwen3vl2b_r16_ep1/int4/onnx/visual")),
}
SPECS["molmo2_v16k"]["visual_files"] = SPECS["molmo2"]["visual_files"]


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.realpath(src), dst)


def main():
    s = SPECS[sys.argv[1]]
    B = os.path.join(D, s["name"])
    for f in sorted(os.listdir(s["llm"])):
        link(os.path.join(s["llm"], f), os.path.join(B, "llm", f))
    if "visual_files" in s:
        for f, src in s["visual_files"].items():
            link(src, os.path.join(B, "visual", f))
    else:
        for f in sorted(os.listdir(s["visual_dir"])):
            link(os.path.join(s["visual_dir"], f), os.path.join(B, "visual", f))
    rows, tot = [], 0
    for sub in ["llm", "visual"]:
        for f in sorted(os.listdir(os.path.join(B, sub))):
            p = os.path.join(B, sub, f)
            n = os.path.getsize(os.path.realpath(p))
            rows.append((f"{s['name']}/{sub}/{f}", os.path.realpath(p), n))
            tot += n
    C = os.path.join(D, "common")
    ctot = 0
    for dp, _, fs in os.walk(C):
        for f in fs:
            p = os.path.join(dp, f)
            ctot += os.path.getsize(os.path.realpath(p))
    with open(os.path.join(B, "copy_list.txt"), "w") as fo:
        fo.write("# 묶음 상대경로\t실제 파일\t바이트\n")
        for a, b, n in rows:
            fo.write(f"{a}\t{b}\t{n}\n")
    info = dict(bundle=s["name"], model_bytes=tot, model_GB=round(tot / 1e9, 3), model_GiB=round(tot / 2**30, 3),
                common_bytes=ctot, common_GB=round(ctot / 1e9, 3), total_GB=round((tot + ctot) / 1e9, 3),
                files=[dict(path=a, bytes=n) for a, _, n in rows])
    json.dump(info, open(os.path.join(B, "bundle_info.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in info.items() if k != "files"}, ensure_ascii=False))
    for a, _, n in rows:
        print(f"  {a}  {n / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
