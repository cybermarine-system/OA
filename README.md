# 원격탐사 VQA 비전-언어 모델의 Jetson Orin Nano 8GB 온디바이스 배포

원격탐사 영상 질의응답(VRSBench VQA)용 비전-언어 모델 세 개를 Jetson Orin Nano 8GB(Super)에 올리기 위해 쓴 코드와, 논문 표를 다시 계산하는 데 필요한 최소 결과 데이터.

- 모델: Molmo2-4B, Qwen3-VL-4B-Instruct, Qwen3-VL-2B-Instruct
- 적용 기법
  1. LoRA 미세조정 (rank 16, 1 epoch, VRSBench train). Molmo2는 Edge-LLM 제약으로 causal attention으로 학습
  2. INT4 AWQ 가중치 양자화 (W4A16, group 128, VRSBench 보정 텍스트 512개, TensorRT Edge-LLM)
  3. 출력 어휘 축소 151,936 → 16,384 (v16k, VRSBench train의 답·질문 토큰 기준 lm_head 행 선택)
  4. TensorRT 엔진 + 요청 100개 단위 분할 실행 + 시각 입력 고정 프로파일
- 평가: A100에서 정확도(test 전체 37,409문항, 층화 표본 2,001문항), Orin에서 지연·메모리·전력·정확도(2,000문항)

## 주요 결과

A100, test 전체 37,409문항 EM (%), 95% CI는 `results/tables/results_final.csv`

| 모델 | FP16 (LoRA 병합) | INT4 | INT4 + v16k |
|---|---|---|---|
| Molmo2-4B | 71.26 | 70.96 | 70.96 |
| Qwen3-VL-4B | 70.73 | 70.12 | 70.06 |
| Qwen3-VL-2B | 69.57 | 68.53 | 68.53 |

Orin Nano 8GB (전원 25W, CPU 1,344 MHz 고정, INT4 + v16k, 배치 1, 최대 16토큰, 5회 × 500질의)

| 모델 | 지연 평균 / p95 (ms) | RAM 유휴 → 피크 (MB) | 질의당 에너지 (J) | 2,000문항 EM, Orin / A100 (%) |
|---|---|---|---|---|
| Molmo2-4B | 1,312 / 1,359 | 1,283 → 5,413 | 28.3 | 70.55 / 70.60 |
| Qwen3-VL-4B | 424 / 468 | 1,494 → 5,549 | 9.0 | 71.05 / 71.05 |
| Qwen3-VL-2B | 236 / 257 | 1,506 → 4,073 | 4.9 | 68.25 / 68.40 |

- 어휘 축소 전후(Orin, INT4 전체 어휘 대 v16k): 프로세스 최대 RSS 516~689 MB 감소, Qwen3-VL-2B 지연 17.5 ms(6.9%) 감소, 보드 답 일치 99.3~99.95% (`results/tables/compare_vocab.json`). Qwen3-VL-4B 전체 어휘 측정은 전원 모드가 MAXN_SUPER, CPU 클럭 비고정이라 지연·전력·에너지를 비교에서 뺌
- Qwen3-VL-2B FP16(+ v16k)은 Orin에서 언어 엔진 빌드가 GPU 메모리 할당 실패로 탑재 불가 (`results/orin/qwen3vl2b_r16_ep1_fp16_v16k/`)
- 언어 엔진 크기는 TensorRT 빌드 로그의 Total Weights Memory. 입력 임베딩(Molmo2와 Qwen3-VL-4B 0.778 GB, Qwen3-VL-2B 0.622 GB)은 엔진 밖 별도 파일이라 포함하지 않음

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| `train/` | LoRA 학습(Molmo2, Qwen3-VL), 어댑터 병합, 학습 그림 스크립트 |
| `quantize_export/` | INT4 AWQ 양자화 래퍼, 보정 집합·어휘 맵 생성, 양자화·ONNX export 실행 스크립트 |
| `quantize_export/molmo2_conversion/` | Molmo2 언어부를 Qwen3 형식으로 바꾸기, 시각부 ONNX export, 변환 검증 |
| `quantize_export/fp16_v16k_2b/` | Qwen3-VL-2B FP16 + v16k export와 A100 기준 측정 |
| `edgellm_patches/` | TensorRT Edge-LLM 1231d64 에 적용한 패치 3개 (Molmo2 runner, 호스트 전처리, 요청별 시간 기록) |
| `eval/` | 평가 집합 생성, 채점, 대응 비교(McNemar, 95% CI), A100 test 전체·지연 측정 |
| `deploy_orin/` | Orin 묶음 생성, 요청 분할, 측정(`measure_orin_v2.sh`), 결과 요약 |
| `deploy_orin/board/` | 보드에서 실행한 스크립트(클럭 고정, 시각 엔진 빌드, 측정 연쇄 실행, 묶음 설치) |
| `deploy_orin/demo/` | 보드 질의 데모 `ask.sh` |
| `analysis/` | 결과 표 생성(`make_tables.py`, `compare_vocab.py`)과 원시 로그 분석(`analyze_orin_v2.py`, `thermal_clock_check.py`) |
| `data/` | 평가 표본 qid 목록, 보정 텍스트 512개, v16k 어휘 맵 |
| `results/` | 표 재계산에 필요한 결과 (아래) |
| `environment/` | A100·Orin 환경 기록, pip freeze, Edge-LLM 커밋 |

`results/`

| 경로 | 내용 |
|---|---|
| `tables/` | `results_final.csv`(주 결과표), `type_accuracy.csv`(유형별 EM), `latency_per_query.csv`(보드 질의별 지연), `compare_vocab.json`(어휘 축소 전후) |
| `a100/test_full/` | A100 test 전체 채점 결과 9개 (3모델 × FP16·INT4·INT4+v16k, gzip) |
| `a100/subset2001/` | A100 2,001문항 채점 결과 9개 |
| `a100/bundle_check_1024/` | Orin과 같은 엔진 옵션(1024/1152/1)의 A100 사전 확인 결과 (전체 어휘 비교 기준) |
| `a100/ref_fp16_v16k_2b/` | Qwen3-VL-2B FP16 + v16k의 A100 기준 결과 |
| `a100/latency/summary.json` | A100 지연·GPU 메모리 요약 |
| `orin/<모델>_int4_{v16k,full}/` | 보드 측정 요약(`summary.json`), 엔진 빌드 로그, 측정 환경(`env.txt`), 구간 시각(`marks.txt`) |
| `orin/compare/` | 보드 2,000문항 채점 결과와 A100 대비 비교 |
| `orin/analysis/` | 원시 로그에서 계산한 지연 분해·전력·RAM(`analysis*.json`), 온도·클럭(`thermal_clock*.json`) |
| `training/` | 학습 설정·손실 기록, 병합 정보, 학습 조건 비교용 2,001문항 결과 |
| `figures/` | 학습 관련 그림 |

## 환경

| 장치 | 내용 |
|---|---|
| A100 서버 | A100 80GB PCIe, 드라이버 595.80, Ubuntu 24.04.4, Python 3.12.3 (`environment/a100_env.txt`) |
| 학습·HF 평가 | torch 2.11.0+cu128, transformers 4.57.6, peft 0.21.2 (`environment/pip_freeze_molmo2_venv.txt`) |
| 양자화·export | tensorrt-edgellm 0.11.0, nvidia-modelopt 0.45.0, tensorrt_cu13 11.4.0.106 (`environment/pip_freeze_venv-edgellm.txt`) |
| A100 C++ 빌드 | TensorRT Edge-LLM 1231d64 + `edgellm_patches/`, CUDA 13.0, TensorRT 11.4 |
| Orin | Jetson Orin Nano 8GB Super 개발자 키트, L4T R39.2 (JetPack 7.2.1), CUDA 13.2, TensorRT 10.16.2 (`environment/board_env_20261119.txt`) |

기반 모델 (Hugging Face revision)
- `allenai/Molmo2-4B` @ `042abfa7a38879a376cec03d949eff0aefaa0600`
- `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`
- `Qwen/Qwen3-VL-2B-Instruct` @ `89644892e4d85e24eaac8bacfd4f463576704203`

## 표 재계산 (결과 데이터만으로)

Python 3 표준 라이브러리만 씀. 저장소 루트에서:

```bash
python3 analysis/make_tables.py      # results/tables/results_final.csv, type_accuracy.csv
python3 analysis/compare_vocab.py    # results/tables/compare_vocab.json
```

포함된 `results/tables/` 파일과 바이트 단위로 같게 나오는 것을 확인함. `latency_per_query.csv`와 `results/orin/analysis/*.json`은 저장소에서 뺀 보드 원시 로그(아래)로 계산한 값이라 그대로 넣었음.

## 전체 재현 순서

스크립트는 원 작업 폴더 구조를 전제로 하고 경로가 절대 경로로 적혀 있음(공개용으로 `/home/user/` 로 바꿔 둠). 다시 돌리려면 아래 대응대로 배치하거나 각 스크립트의 경로 변수를 고쳐야 함. 각 스크립트 사용법은 파일 머리 주석에 있음.

| 이 저장소 | 원 작업 폴더 (`/home/user/ondeviceAI/` 기준) |
|---|---|
| `train/` | `lora/scripts/` (`run_*.sh` 는 `lora/`) |
| `quantize_export/` | `quant/scripts/` (`run_*.sh` 는 `quant/`) |
| `quantize_export/molmo2_conversion/`, `eval/eval_vrsbench_qwen3vl.py`, `eval/edgellm_out_to_jsonl.py` | `edgellm_check/scripts/` |
| `eval/eval_vrsbench.py`, `eval/compare_runs.py`, `eval/vrsbench_stats.py` | `molmo2/scripts/` |
| `eval/` 나머지 | `quant/scripts/` (`run_*.sh`, `bundle_check_v16k.sh`, `compare_full_v16k.sh` 는 `quant/`) |
| `deploy_orin/make_chunks.py`, `make_orin_bundle.py` | `deploy/` |
| `deploy_orin/measure_orin_v2.sh`, `merge_outputs.py`, `summarize_orin_v2.py` | `deploy/orin/common/` (보드에서는 `~/vrs_bundle_v2/common/`) |
| `deploy_orin/board/` | 보드 `~/v2/` |

1. **데이터**: Hugging Face `xiang709/VRSBench` 에서 `VRSBench_train.json`, `VRSBench_EVAL_vqa.json`, `Images_train.zip`, `Images_val.zip` 을 받아 `molmo2/data/VRSBench/` 에 풂. 평가 집합은 `eval/make_eval_sets.py`(test 전체, 2,001 층화 표본의 앞 2,000개, 효율 측정 500개, 워밍업 10개, 모두 seed 0). 이 저장소의 `data/splits/*_samples.jsonl` 이 그 결과 qid 목록
2. **학습**: `train/train_lora_vrsbench.py`(Molmo2, 기본 causal), `train/train_lora_qwen3vl.py`(Qwen3-VL) → `train/merge_lora_to_hf.py` 로 병합. 학습 설정과 손실 기록은 `results/training/runs/*/`
3. **양자화·export**: `quantize_export/make_calib512.py`(보정 텍스트, `data/calib/`), `make_vocab_map.py`(v16k 어휘 맵, `data/vocab_v16k/`). Molmo2는 `molmo2_conversion/convert_molmo2_llm_to_qwen3.py` 로 언어부를 Qwen3 형식으로 바꾼 뒤, 세 모델 모두 `edgellm_quantize_vrs.py`(INT4 AWQ)와 Edge-LLM export. 실행 순서는 `run_molmo2_int4*.sh`, `run_qwen3vl_post.sh`, `run_qwen3vl_v16k.sh`
4. **A100 평가**: Edge-LLM 1231d64 에 `edgellm_patches/` 를 `git apply` 로 차례로 적용해 빌드. `eval/run_full_test.sh`(test 전체), `eval/run_latency_a100.sh`(지연), `eval/edgellm_out_to_jsonl.py`(채점), `eval/compare_pair.py`(대응 비교). Orin 엔진 옵션 사전 확인은 `eval/bundle_check_v16k.sh`
5. **Orin 측정**: 보드에서 같은 Edge-LLM + 패치를 빌드. 묶음은 `deploy_orin/make_orin_bundle.py` 로 만들고 보드에서 `board/setup_vrs_bundle_v2.sh` 로 설치. 엔진은 `llm_build`(언어, 옵션 1024/1152/1)와 `board/build_visual.sh`(시각). 측정은 전원 모드를 25W로 두고 스왑을 끈 상태에서 `board/fix_clocks.sh`(25W 확인, `jetson_clocks`와 팬 최대, CPU 1,344 MHz·GPU 918 MHz 확인)를 실행한 뒤 `board/measure.sh <모델>` (내부에서 `measure_orin_v2.sh`, 결과 요약은 `summarize_orin_v2.py`). 여러 모델 연쇄 실행은 `board/chain_orin.sh`, `board/chain_full.sh`
6. **분석**: `analysis/analyze_orin_v2.py [v16k|full]`, `analysis/thermal_clock_check.py [v16k|full]`(원시 로그 필요) → `analysis/make_tables.py`, `analysis/compare_vocab.py`
7. **데모**: 보드에서 `bash deploy_orin/demo/ask.sh <엔진> <이미지파일명> "<질문>" [free]` (엔진: `molmo2`, `qwen4b`, `qwen2b`, 전체 어휘는 `_full` 접미사)

## 저장소에 넣지 않은 것

- 모델 가중치, LoRA 어댑터, AWQ 체크포인트, ONNX, TensorRT 엔진 (위 순서로 다시 만듦)
- VRSBench 이미지와 주석 (Hugging Face에서 받음)
- 보드 원시 로그: tegrastats 100 ms 기록, 분할 실행별 출력·로그·프로파일(`eff_rep*_c*`, `cons_c*`). `results/orin/analysis/*.json` 과 `results/tables/latency_per_query.csv` 가 이 로그에서 계산한 값
- 최종 설정에 쓰지 않은 중간 실험 결과(학습 단계별 평가, HF 경로 평가 등)

## 단위

RAM·RSS MB(2^20 B, tegrastats·free 표기), A100 GPU 메모리 MiB, 엔진·파일 크기 GB(10^9 B), 지연 ms, 전력 W, 에너지 J, 정확도 %, 차이 %p

## 데이터셋과 인용

VRSBench는 CC-BY-4.0. 원 영상은 DOTA, DIOR 기반이므로 원 데이터셋 조건도 함께 따름.

```bibtex
@inproceedings{li2024vrsbench,
  title     = {VRSBench: A Versatile Vision-Language Benchmark Dataset for Remote Sensing Image Understanding},
  author    = {Li, Xiang and Ding, Jian and Elhoseiny, Mohamed},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year      = {2024}
}
```

기반 모델과 TensorRT Edge-LLM은 각 원저작자의 라이선스를 따름. 이 저장소 코드의 라이선스는 미정.
