"""tensorrt-edgellm-quantize를 VRSBench 보정 텍스트(quant/calib/vrsbench_calib512.jsonl)로 실행하는 래퍼.

- Edge-LLM 사용자 데이터셋 양식(custom_dataset_template.py)대로 @register("text", 이름) 생성기를 등록한 뒤
  원래 CLI(tensorrt_edgellm.scripts.quantize:main)를 그대로 호출함. 설치된 패키지 파일은 고치지 않음
- 등록 이름: vrsbench_calib512_molmo2 / vrsbench_calib512_qwen3vl4b / vrsbench_calib512_qwen3vl2b (각 모델 채팅 템플릿 적용본)
- --force_text_calib: 시각부가 있는 모델(Qwen3-VL)에서 이 버전 도구는 AWQ면 이미지 보정(_is_image_blind_calibration)으로
  자동 전환함. 전략 문서(텍스트 보정, 세 모델 동일)를 따르기 위해 이 판정을 False로 고정함
- 보정 텍스트 해시와 실제 사용 개수를 <output_dir>/calib_info.json에 기록

usage (.venv-edgellm, LD_LIBRARY_PATH=edgellm_libs, 소스 폴더 밖에서):
  .venv-edgellm/bin/python quant/scripts/edgellm_quantize_vrs.py [--force_text_calib] -- llm --model_dir <hf> \
      --output_dir <out> --quantization int4_awq --text_dataset vrsbench_calib512_molmo2
"""
import hashlib
import json
import os
import sys

CALIB = "/home/user/ondeviceAI/quant/calib/vrsbench_calib512.jsonl"

from tensorrt_edgellm.quantization.datasets import register  # noqa: E402

_used = {}


def _make(model_key):
    def gen():
        n = 0
        for line in open(CALIB):
            t = json.loads(line)["text_" + model_key]
            if t:
                n += 1
                _used[model_key] = n
                yield t
    gen.__name__ = f"vrsbench_calib512_{model_key}"
    return gen


for _k in ("molmo2", "qwen3vl4b", "qwen3vl2b"):
    register("text", f"vrsbench_calib512_{_k}")(_make(_k))


def main():
    argv = sys.argv[1:]
    force = "--force_text_calib" in argv
    argv = [a for a in argv if a != "--force_text_calib"]
    if argv and argv[0] == "--":
        argv = argv[1:]
    if force:
        import tensorrt_edgellm.quantization.quantize as q
        q._is_image_blind_calibration = lambda model, cfg: False
        print("[vrs] 이미지 보정 자동 전환 끔: 텍스트 보정 강제", flush=True)
    out = argv[argv.index("--output_dir") + 1] if "--output_dir" in argv else None
    sys.argv = ["tensorrt-edgellm-quantize"] + argv
    from tensorrt_edgellm.scripts.quantize import main as qmain
    qmain()
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(dict(calib_file=CALIB, sha256=hashlib.sha256(open(CALIB, "rb").read()).hexdigest(),
                       used=_used, force_text_calib=force, argv=argv),
                  open(os.path.join(out, "calib_info.json"), "w"), indent=1, ensure_ascii=False)
        print("[vrs] calib_info", _used, flush=True)


if __name__ == "__main__":
    main()
