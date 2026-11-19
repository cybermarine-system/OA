"""변환 검증: 같은 텍스트 입력에서 HF Molmo2-4B와 변환된 Qwen3ForCausalLM의 logits·greedy 출력 비교."""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

CKPT, REV = "allenai/Molmo2-4B", "042abfa7a38879a376cec03d949eff0aefaa0600"
conv_dir, out = sys.argv[1], sys.argv[2]
PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "How many legs does a spider have? Answer with a number.",
    "Name three types of ships used in a harbor.",
]

proc = AutoProcessor.from_pretrained(CKPT, revision=REV, trust_remote_code=True)
molmo = AutoModelForImageTextToText.from_pretrained(
    CKPT, revision=REV, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda").eval()
qwen = AutoModelForCausalLM.from_pretrained(conv_dir, dtype=torch.bfloat16, device_map="cuda").eval()

rows = []
for p in PROMPTS:
    msgs = [{"role": "user", "content": [{"type": "text", "text": p}]}]
    inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
                                   return_dict=True)
    ids = inp["input_ids"].cuda()
    with torch.inference_mode():
        lm = molmo(input_ids=ids).logits[0, :, :151936].float()
        lq = qwen(input_ids=ids).logits[0, :, :151936].float()
        gm = molmo.generate(input_ids=ids, max_new_tokens=48, do_sample=False)[0, ids.shape[1]:]
        gq = qwen.generate(input_ids=ids, max_new_tokens=48, do_sample=False)[0, ids.shape[1]:]
    r = dict(prompt=p, prompt_text=proc.tokenizer.decode(ids[0]), n_tokens=ids.shape[1],
             max_abs_logit_diff=(lm - lq).abs().max().item(),
             argmax_match=(lm.argmax(-1) == lq.argmax(-1)).float().mean().item(),
             molmo_out=proc.tokenizer.decode(gm, skip_special_tokens=True),
             qwen_out=proc.tokenizer.decode(gq, skip_special_tokens=True),
             gen_ids_equal=torch.equal(gm, gq), molmo_ids=gm.tolist())
    rows.append(r)
    print(json.dumps({k: v for k, v in r.items() if k != "molmo_ids"}, ensure_ascii=False))
json.dump(rows, open(out, "w"), ensure_ascii=False, indent=1)
