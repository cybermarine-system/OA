"""INT4 체크포인트 산출물 목록: 양자화 설정, 실제 INT4로 바뀐 모듈 목록(가중치 dtype·scale 존재로 판정), FP16 유지 모듈, 파일 해시.

전략 문서 3-가 "산출물: INT4 가중치·스케일, 양자화 설정 파일, 실제 변환 모듈 목록 및 해시"를 채우기 위함

usage (.venv-edgellm 또는 molmo2/.venv): python quant/scripts/quant_manifest.py <int4_awq 폴더> [<onnx 폴더>]
  → <int4_awq>/manifest.json
"""
import collections
import hashlib
import json
import os
import re
import sys

from safetensors import safe_open


def sha256(p, bs=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    d = sys.argv[1]
    extra = sys.argv[2:]
    quant, other = collections.Counter(), collections.Counter()
    qmods, n_tensors = set(), 0
    files = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
    dt = collections.Counter()
    for f in files:
        with safe_open(os.path.join(d, f), "pt") as s:
            keys = list(s.keys())
            n_tensors += len(keys)
            ks = set(keys)
            for k in keys:
                if k.endswith(".weight_scale") or k.endswith(".weight_scale_2"):
                    qmods.add(k.rsplit(".", 1)[0])
            for k in keys:
                if k.endswith(".weight"):
                    mod = k[:-7]
                    pat = re.sub(r"\.\d+\.", ".N.", mod)
                    t = s.get_slice(k).get_dtype()
                    dt[(pat, t)] += 1
                    (quant if mod in qmods else other)[pat] += 1
    man = dict(
        dir=os.path.abspath(d),
        hf_quant_config=json.load(open(os.path.join(d, "hf_quant_config.json"))) if os.path.exists(os.path.join(d, "hf_quant_config.json")) else None,
        calib_info=json.load(open(os.path.join(d, "calib_info.json"))) if os.path.exists(os.path.join(d, "calib_info.json")) else None,
        n_tensors=n_tensors, n_quantized_modules=len(qmods),
        quantized_module_patterns=dict(quant), non_quantized_weight_patterns=dict(other),
        weight_dtypes={f"{p} [{t}]": n for (p, t), n in sorted(dt.items())},
        files={},
    )
    for root in [d] + extra:
        for dp, _, fs in os.walk(root):
            for f in sorted(fs):
                p = os.path.join(dp, f)
                if f.endswith((".safetensors", ".json", ".onnx", ".data", ".jinja")):
                    man["files"][os.path.relpath(p, os.path.dirname(os.path.abspath(d)))] = dict(
                        bytes=os.path.getsize(p), sha256=sha256(p))
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: man[k] for k in ["n_tensors", "n_quantized_modules", "quantized_module_patterns",
                                          "non_quantized_weight_patterns"]}, indent=1))


if __name__ == "__main__":
    main()
