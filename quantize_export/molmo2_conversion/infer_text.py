"""Edge-LLM 엔진으로 텍스트 프롬프트 greedy 추론 (설치된 휠의 native LLMRuntime 사용)."""
import argparse
import json
import time

from tensorrt_edgellm import runtime

PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "How many legs does a spider have? Answer with a number.",
    "Name three types of ships used in a harbor.",
]

ap = argparse.ArgumentParser()
ap.add_argument("--engine-dir", required=True)
ap.add_argument("--checkpoint-dir", required=True)
ap.add_argument("--max-tokens", type=int, default=48)
ap.add_argument("--out", required=True)
args = ap.parse_args()

native = runtime.load()
t0 = time.time()
llm = native.LLMRuntime(engine_dir=args.engine_dir, checkpoint_dir=args.checkpoint_dir)
print(f"runtime init {time.time() - t0:.1f}s")

rows = []
for p in PROMPTS:
    req = native.create_generation_request([[native.create_text_message("user", p)]],
                                           temperature=0.0, max_generate_length=args.max_tokens)
    t0 = time.time()
    resp = llm.handle_request(req)
    dt = time.time() - t0
    rows.append(dict(prompt=p, output=resp.output_texts[0], output_ids=list(resp.output_ids[0]), sec=round(dt, 3)))
    print(f"[Q] {p}\n[A] {resp.output_texts[0]!r} ({len(resp.output_ids[0])} tok, {dt:.2f}s)")
json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=1)
