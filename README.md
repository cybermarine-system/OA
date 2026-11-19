# Jetson Orin Nano에서 원격탐사 VQA 모델 실행하기

VRSBench 영상의 질문에 답하는 비전 언어 모델 3종을 Jetson Orin Nano 8GB에 배포하고 평가한 코드와 결과 데이터입니다. LoRA 학습부터 INT4 양자화, TensorRT 엔진 빌드, 정확도와 효율 측정까지의 과정을 담았습니다.

| 항목 | 내용 |
| --- | --- |
| 모델 | Molmo2-4B, Qwen3-VL-4B-Instruct, Qwen3-VL-2B-Instruct |
| 데이터셋 | VRSBench VQA |
| 서버 평가 | A100에서 전체 test 37,409문항과 층화 표본 2,001문항 |
| 보드 평가 | Jetson Orin Nano 8GB Super에서 지연, 메모리, 전력, 정확도 2,000문항 |
| 최종 배포 설정 | LoRA 병합, INT4 AWQ, 출력 어휘 16,384개(v16k), TensorRT Edge-LLM |

## 주요 결과

### A100 정확도

전체 test 37,409문항의 Exact Match(EM, %). 95% 신뢰구간을 포함한 원본 표는 [`results/tables/results_final.csv`](results/tables/results_final.csv)에 있습니다.

| 모델 | FP16 (LoRA 병합) | INT4 | INT4 + v16k |
| --- | ---: | ---: | ---: |
| Molmo2-4B | 71.26 | 70.96 | 70.96 |
| Qwen3-VL-4B | 70.73 | 70.12 | 70.06 |
| Qwen3-VL-2B | 69.57 | 68.53 | 68.53 |

### Orin Nano 실행 결과

25W 모드, CPU 1,344 MHz 고정, INT4 + v16k, 배치 1, 최대 출력 16토큰에서 측정했습니다. 지연과 에너지는 500질의씩 5회 반복한 결과입니다.

| 모델 | 지연 평균 / p95 (ms) | RAM 유휴 → 피크 (MB) | 질의당 에너지 (J) | 2,000문항 EM: Orin / A100 (%) |
| --- | ---: | ---: | ---: | ---: |
| Molmo2-4B | 1,312 / 1,359 | 1,283 → 5,413 | 28.3 | 70.55 / 70.60 |
| Qwen3-VL-4B | 424 / 468 | 1,494 → 5,549 | 9.0 | 71.05 / 71.05 |
| Qwen3-VL-2B | 236 / 257 | 1,506 → 4,073 | 4.9 | 68.25 / 68.40 |

어휘를 전체 크기에서 v16k로 줄이면 Orin의 프로세스 최대 RSS가 516~689 MB 감소했습니다. Qwen3-VL-2B의 평균 지연은 17.5 ms(6.9%) 줄었고, 두 설정의 보드 답변은 99.3~99.95% 일치했습니다. 자세한 비교는 [`results/tables/compare_vocab.json`](results/tables/compare_vocab.json)에 있습니다. Qwen3-VL-4B의 전체 어휘 실험은 전원 모드가 MAXN_SUPER이고 CPU 클럭이 고정되지 않아 지연, 전력, 에너지 비교에서 제외했습니다.

Qwen3-VL-2B FP16 + v16k는 Orin에서 언어 엔진을 빌드할 때 GPU 메모리 할당에 실패했습니다. 기록은 `results/orin/qwen3vl2b_r16_ep1_fp16_v16k/`에 있습니다.

## 이 저장소에서 할 수 있는 일

| 단계 | 방법 |
| --- | --- |
| 학습 | VRSBench train으로 LoRA 미세조정(rank 16, 1 epoch) 후 어댑터 병합 |
| 양자화 | 보정 텍스트 512개로 INT4 AWQ 적용(W4A16, group 128) |
| 어휘 축소 | train의 답변과 질문 토큰을 기준으로 `lm_head`의 행을 선택해 출력 어휘를 151,936개에서 16,384개로 축소 |
| 배포 | TensorRT 엔진 빌드, 시각 입력 고정 프로파일 적용, 요청 100개 단위로 분할 실행 |
| 평가 | A100 정확도와 지연, Orin 정확도와 지연, 메모리와 전력 측정 |

Molmo2는 Edge-LLM의 제약에 맞춰 causal attention으로 학습했습니다.

## 결과 표 다시 계산하기

저장소 루트에서 아래 명령을 실행하면 됩니다. Python 3 표준 라이브러리만 사용합니다.

```bash
python3 analysis/make_tables.py    # results_final.csv, type_accuracy.csv
python3 analysis/compare_vocab.py  # compare_vocab.json
```

두 스크립트의 출력이 포함된 `results/tables/` 파일과 바이트 단위로 같은 것을 확인했습니다. `latency_per_query.csv`와 `results/orin/analysis/*.json`은 저장소에 포함하지 않은 보드 원시 로그에서 계산한 결과이므로 재계산 대상에 포함되지 않습니다.

## 전체 실험 재현하기

스크립트에는 원래 작업 폴더를 기준으로 한 절대 경로가 들어 있습니다. 공개본에서는 해당 경로를 `/home/user/`로 바꿨습니다. 아래 폴더 대응에 맞게 파일을 배치하거나 스크립트의 경로 변수를 수정하세요. 각 스크립트의 사용법은 파일 첫머리 주석에 있습니다.

| 이 저장소 | 원 작업 폴더 (`/home/user/ondeviceAI/` 기준) |
| --- | --- |
| `train/` | `lora/scripts/` (`run_*.sh`는 `lora/`) |
| `quantize_export/` | `quant/scripts/` (`run_*.sh`는 `quant/`) |
| `quantize_export/molmo2_conversion/`, `eval/eval_vrsbench_qwen3vl.py`, `eval/edgellm_out_to_jsonl.py` | `edgellm_check/scripts/` |
| `eval/eval_vrsbench.py`, `eval/compare_runs.py`, `eval/vrsbench_stats.py` | `molmo2/scripts/` |
| `eval/`의 나머지 파일 | `quant/scripts/` (`run_*.sh`, `bundle_check_v16k.sh`, `compare_full_v16k.sh`는 `quant/`) |
| `deploy_orin/make_chunks.py`, `make_orin_bundle.py` | `deploy/` |
| `deploy_orin/measure_orin_v2.sh`, `merge_outputs.py`, `summarize_orin_v2.py` | `deploy/orin/common/` (보드에서는 `~/vrs_bundle_v2/common/`) |
| `deploy_orin/board/` | 보드의 `~/v2/` |

1. **데이터 준비:** Hugging Face의 `xiang709/VRSBench`에서 `VRSBench_train.json`, `VRSBench_EVAL_vqa.json`, `Images_train.zip`, `Images_val.zip`을 받아 `molmo2/data/VRSBench/`에 풉니다. `eval/make_eval_sets.py`로 전체 test, 2,001문항 층화 표본의 앞 2,000개, 효율 측정용 500개, 워밍업용 10개를 생성합니다. 모두 seed 0을 사용하며, 결과 qid 목록은 `data/splits/*_samples.jsonl`에 있습니다.
2. **학습:** Molmo2는 `train/train_lora_vrsbench.py`(기본 causal), Qwen3-VL은 `train/train_lora_qwen3vl.py`로 학습합니다. `train/merge_lora_to_hf.py`로 어댑터를 병합합니다. 학습 설정과 손실 기록은 `results/training/runs/*/`에 있습니다.
3. **양자화 및 export:** `quantize_export/make_calib512.py`로 보정 텍스트를, `make_vocab_map.py`로 v16k 어휘 맵을 만듭니다. 결과는 각각 `data/calib/`, `data/vocab_v16k/`에 있습니다. Molmo2 언어부는 `molmo2_conversion/convert_molmo2_llm_to_qwen3.py`로 Qwen3 형식으로 변환합니다. 세 모델 모두 `edgellm_quantize_vrs.py`로 INT4 AWQ를 적용하고 Edge-LLM으로 export합니다. 실행 순서는 `run_molmo2_int4*.sh`, `run_qwen3vl_post.sh`, `run_qwen3vl_v16k.sh`를 참고하세요.
4. **A100 평가:** TensorRT Edge-LLM 커밋 `1231d64`에 `edgellm_patches/`의 패치 3개를 `git apply`로 순서대로 적용한 뒤 빌드합니다. `eval/run_full_test.sh`로 전체 test를 평가하고, `eval/run_latency_a100.sh`로 지연을 측정합니다. 채점에는 `eval/edgellm_out_to_jsonl.py`, 대응 비교에는 `eval/compare_pair.py`를 사용합니다. Orin 엔진 옵션은 `eval/bundle_check_v16k.sh`에서 먼저 확인합니다.
5. **Orin 측정:** 보드에서도 같은 Edge-LLM과 패치를 빌드합니다. `deploy_orin/make_orin_bundle.py`로 묶음을 만들고 `board/setup_vrs_bundle_v2.sh`로 설치합니다. 언어 엔진은 `llm_build`(옵션 1024/1152/1), 시각 엔진은 `board/build_visual.sh`로 빌드합니다. 25W 모드에서 스왑을 끈 뒤 `board/fix_clocks.sh`를 실행해 `jetson_clocks`, 팬 최대 속도, CPU 1,344 MHz, GPU 918 MHz를 확인합니다. 이어 `board/measure.sh <모델>`을 실행합니다. 내부 측정 스크립트는 `measure_orin_v2.sh`, 요약 스크립트는 `summarize_orin_v2.py`입니다. 여러 모델을 연속 실행하려면 `board/chain_orin.sh` 또는 `board/chain_full.sh`를 사용합니다.
6. **분석:** 원시 로그가 있으면 `analysis/analyze_orin_v2.py [v16k|full]`, `analysis/thermal_clock_check.py [v16k|full]`로 보드 로그를 분석합니다. 이후 `analysis/make_tables.py`와 `analysis/compare_vocab.py`로 표를 만듭니다.
7. **데모:** 보드에서 다음 형식으로 질문합니다. 엔진 이름은 `molmo2`, `qwen4b`, `qwen2b` 중 하나이며 전체 어휘 엔진에는 `_full`을 붙입니다.

   ```bash
   bash deploy_orin/demo/ask.sh <엔진> <이미지파일명> "<질문>" [free]
   ```

## 폴더 안내

| 폴더 | 내용 |
| --- | --- |
| `train/` | LoRA 학습, 어댑터 병합, 학습 그림 생성 |
| `quantize_export/` | INT4 AWQ, 보정 집합과 어휘 맵 생성, ONNX export |
| `quantize_export/molmo2_conversion/` | Molmo2 언어부 형식 변환, 시각부 ONNX export와 변환 검증 |
| `quantize_export/fp16_v16k_2b/` | Qwen3-VL-2B FP16 + v16k export와 A100 기준 측정 |
| `edgellm_patches/` | Molmo2 runner, 호스트 전처리, 요청별 시간 기록을 위한 패치 3개 |
| `eval/` | 평가 집합 생성, 채점, McNemar 대응 비교, 95% 신뢰구간, A100 측정 |
| `deploy_orin/` | 보드용 묶음 생성, 요청 분할, 측정 및 요약 |
| `deploy_orin/board/` | 클럭 고정, 시각 엔진 빌드, 연속 측정과 설치 |
| `deploy_orin/demo/` | 보드 질의 데모 `ask.sh` |
| `analysis/` | 결과 표 생성, 원시 로그와 온도 및 클럭 분석 |
| `data/` | 평가 표본 qid 목록, 보정 텍스트 512개, v16k 어휘 맵 |
| `results/` | 표 재계산에 필요한 결과 데이터 |
| `environment/` | A100 및 Orin 환경, pip freeze, Edge-LLM 커밋 기록 |

### `results/` 구성

| 경로 | 내용 |
| --- | --- |
| `tables/` | `results_final.csv`(주 결과), `type_accuracy.csv`(유형별 EM), `latency_per_query.csv`(보드 질의별 지연), `compare_vocab.json`(어휘 비교) |
| `a100/test_full/` | 모델 3종과 설정 3종의 전체 test 채점 결과 9개(gzip) |
| `a100/subset2001/` | 2,001문항 채점 결과 9개 |
| `a100/bundle_check_1024/` | Orin과 같은 엔진 옵션(1024/1152/1)의 A100 사전 확인 결과 |
| `a100/ref_fp16_v16k_2b/` | Qwen3-VL-2B FP16 + v16k의 A100 기준 결과 |
| `a100/latency/summary.json` | A100 지연과 GPU 메모리 요약 |
| `orin/<모델>_int4_{v16k,full}/` | 보드 측정 `summary.json`, 엔진 빌드 로그, `env.txt`, `marks.txt` |
| `orin/compare/` | 보드 2,000문항 채점 결과와 A100 비교 |
| `orin/analysis/` | 원시 로그 분석 `analysis*.json`, 온도 및 클럭 분석 `thermal_clock*.json` |
| `training/` | 학습 설정과 손실 기록, 병합 정보, 2,001문항 학습 조건 비교 |
| `figures/` | 학습 관련 그림 |

## 실행 환경

| 구분 | 사양과 버전 |
| --- | --- |
| A100 서버 | A100 80GB PCIe, 드라이버 595.80, Ubuntu 24.04.4, Python 3.12.3 (`environment/a100_env.txt`) |
| 학습 및 HF 평가 | torch 2.11.0+cu128, transformers 4.57.6, peft 0.21.2 (`environment/pip_freeze_molmo2_venv.txt`) |
| 양자화 및 export | tensorrt-edgellm 0.11.0, nvidia-modelopt 0.45.0, tensorrt_cu13 11.4.0.106 (`environment/pip_freeze_venv-edgellm.txt`) |
| A100 C++ 빌드 | TensorRT Edge-LLM `1231d64`와 `edgellm_patches/`, CUDA 13.0, TensorRT 11.4 |
| Orin | Jetson Orin Nano 8GB Super 개발자 키트, L4T R39.2 (JetPack 7.2.1), CUDA 13.2, TensorRT 10.16.2 (`environment/board_env_20261119.txt`) |

기반 모델의 Hugging Face revision은 다음과 같습니다.

| 모델 | Revision |
| --- | --- |
| `allenai/Molmo2-4B` | `042abfa7a38879a376cec03d949eff0aefaa0600` |
| `Qwen/Qwen3-VL-4B-Instruct` | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| `Qwen/Qwen3-VL-2B-Instruct` | `89644892e4d85e24eaac8bacfd4f463576704203` |

## 포함하지 않은 파일과 측정 기준

모델 가중치, LoRA 어댑터, AWQ 체크포인트, ONNX 파일, TensorRT 엔진은 저장소에 포함하지 않았습니다. 위 재현 순서에 따라 생성해야 합니다. VRSBench 이미지와 주석도 Hugging Face에서 별도로 받아야 합니다. 최종 설정에 쓰지 않은 학습 단계별 평가와 HF 경로 평가 등 중간 실험 결과 역시 포함하지 않았습니다.

보드의 `tegrastats` 100 ms 기록, 분할 실행별 출력과 프로파일(`eff_rep*_c*`, `cons_c*`) 같은 원시 로그도 포함하지 않았습니다. `results/orin/analysis/*.json`과 `results/tables/latency_per_query.csv`는 해당 로그에서 계산한 값입니다.

RAM과 RSS의 MB는 2²⁰ B(`tegrastats`, `free` 표기), A100 GPU 메모리는 MiB, 엔진과 파일 크기의 GB는 10⁹ B 기준입니다. 지연은 ms, 전력은 W, 에너지는 J, 정확도는 %, 정확도 차이는 %p로 표기합니다. 언어 엔진 크기는 TensorRT 빌드 로그의 `Total Weights Memory`입니다. 엔진 밖에 별도 파일로 저장되는 입력 임베딩은 포함하지 않습니다(Molmo2와 Qwen3-VL-4B 각 0.778 GB, Qwen3-VL-2B 0.622 GB).

## 데이터셋과 라이선스

VRSBench의 라이선스는 CC-BY-4.0입니다. 원 영상이 DOTA와 DIOR를 기반으로 하므로 해당 데이터셋의 조건도 함께 따라야 합니다.

```bibtex
@inproceedings{li2024vrsbench,
  title     = {VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding},
  author    = {Li, Xiang and Ding, Jian and Elhoseiny, Mohamed},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year      = {2024}
}
```

기반 모델과 TensorRT Edge-LLM에는 각 원저작자의 라이선스가 적용됩니다. 이 저장소 코드의 라이선스는 아직 정해지지 않았습니다.
