"""Edge-LLM llm_inference 출력(JSON)을 eval_vrsbench.py 결과와 같은 jsonl 형식으로 변환·채점.

compare_runs.py(McNemar)와 summarize_runs.py에서 HF 결과와 같은 방식으로 다루기 위한 것.
samples: 요청과 같은 순서의 문항 메타데이터 jsonl (qid, type, gt 필드 필요)
  - Qwen: eval_vrsbench_qwen3vl.py --requests_out가 만든 *_samples.jsonl
  - Molmo2: make_molmo2_inputs.py가 만든 samples.jsonl
--hf_field: samples 안의 HF 예측 필드(예: hf_pred)를 대신 채점해 HF 결과 jsonl을 만들 때 사용

usage: molmo2/.venv/bin/python edgellm_out_to_jsonl.py <samples.jsonl> <edgellm_out.json|-> <out.jsonl> [--hf_field hf_pred]
"""
import argparse
import json
import sys

sys.path.insert(0, "/home/user/ondeviceAI/molmo2/scripts")
from eval_vrsbench import score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("edgellm_out")
    ap.add_argument("out")
    ap.add_argument("--hf_field")
    args = ap.parse_args()
    samples = [json.loads(l) for l in open(args.samples)]
    resp = None if args.hf_field else json.load(open(args.edgellm_out))["responses"]
    if resp is not None:
        assert len(resp) == len(samples), (len(resp), len(samples))
    with open(args.out, "w") as f:
        for i, x in enumerate(samples):
            r = {} if resp is None else resp[i]
            pred = (x[args.hf_field] if args.hf_field else r["output_text"]).strip()
            em, lenient = score(pred, x["gt"], x["type"])
            rec = dict(qid=str(x["qid"]), type=x["type"], gt=x["gt"], pred=pred, em=em, lenient=lenient)
            if r:
                rec.update(n_gen=r.get("generated_token_count"), wall_ms=r.get("request_wall_ms"),
                           finish=r.get("finish_reason"))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("wrote", args.out, len(samples))


if __name__ == "__main__":
    main()
