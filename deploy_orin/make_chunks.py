"""Orin 측정용 요청 분할 파일 생성 (2026-11-16, 메모리 절감 v2).

llm_inference는 요청 파일 전체를 미리 메모리에 올림 (A100 측정 CPU 피크: 10개 1.09 GB, 510개 1.59 GB, 2,001개 3.10 GB).
Orin은 CPU·GPU가 같은 RAM을 쓰므로 한 프로세스에 들어가는 요청을 100개로 묶음.

- 효율: chunks/eff_c1..c5.json = 워밍업 10 + 효율 500을 100개씩 나눈 것 (각 파일 앞 10개는 같은 워밍업, 지연 계산에서 제외)
- 정합성: chunks/cons_c01..c20.json = 정합성 2,000을 순서대로 100개씩 (결과는 merge_outputs.py로 합쳐 원래 순서 복원)
- 요청 내용·생성 설정은 원래 파일(efficiency500/warmup10/consistency2000_requests.json)과 같음

usage: python3 deploy/make_chunks.py
"""
import json
import os

C = "/home/user/ondeviceAI/deploy/orin/common"
N = 100


def dump(head, reqs, path):
    json.dump(dict(head, requests=reqs), open(path, "w"), ensure_ascii=False, indent=1)


def main():
    out = os.path.join(C, "chunks")
    os.makedirs(out, exist_ok=True)
    eff = json.load(open(os.path.join(C, "efficiency500_requests.json")))
    wu = json.load(open(os.path.join(C, "warmup10_requests.json")))
    cons = json.load(open(os.path.join(C, "consistency2000_requests.json")))
    head_e = {k: v for k, v in eff.items() if k != "requests"}
    head_c = {k: v for k, v in cons.items() if k != "requests"}
    assert len(wu["requests"]) == 10 and len(eff["requests"]) == 500 and len(cons["requests"]) == 2000
    for i in range(5):
        dump(head_e, wu["requests"] + eff["requests"][i * N:(i + 1) * N], os.path.join(out, f"eff_c{i + 1}.json"))
    for i in range(20):
        dump(head_c, cons["requests"][i * N:(i + 1) * N], os.path.join(out, f"cons_c{i + 1:02d}.json"))
    info = dict(chunk_size=N, eff_chunks=5, eff_per_chunk="warmup10 + 100", cons_chunks=20,
                note="각 파일은 프로세스 1개로 실행. eff 앞 10개는 지연 계산에서 제외")
    json.dump(info, open(os.path.join(out, "chunks_info.json"), "w"), ensure_ascii=False, indent=1)
    print(info)


if __name__ == "__main__":
    main()
