"""Qwen3-VL-4B/2B-Instruct LoRA 학습 (VRSBench train VQA) — Molmo2 스크립트(train_lora_vrsbench.py)와 같은 조건.

같은 것: 데이터 분할(load_vqa seed 0, 이미지 1% 검증 → 학습 84,973 / 검증 840), 질문 정리(clean_question),
  프롬프트(format_rules, 메시지 순서 text → image), 정답 + <|im_end|>만 손실, 500스텝마다 검증 840쌍,
  저장(adapter_step{N} 500스텝마다, 마지막 adapter, adapter_epoch{N} + train_state.pt), cosine + warmup 50,
  AdamW lr 2e-4 weight decay 0, clipping 1.0, BF16, seed 0, rank 16 / alpha 32 / dropout 0.05
LoRA 대상: 언어 모델 model.language_model.layers.N의 q/k/v/o_proj, gate/up/down_proj만 (시각 인코더·merger·deepstack merger·lm_head 고정)
이미지 전처리: 기존 Qwen 평가(edgellm_check/scripts/eval_vrsbench_qwen3vl.py)와 같은 processor 기본 설정 (512×512 → 영상 토큰 256)
Qwen3-VL은 원래 causal이라 attention 설정 옵션 없음

usage (molmo2/.venv):
  molmo2/.venv/bin/python lora/scripts/train_lora_qwen3vl.py --model <HF snapshot> --out lora/runs/qwen3vl4b_r16 \
      [--bs 16 --grad_accum 2] [--max_steps 20 --val_batches 5]
"""
import argparse
import json
import math
import sys
import time
from os.path import join
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = "/home/user/ondeviceAI"
sys.path.insert(0, join(ROOT, "lora/scripts"))
sys.path.insert(0, join(ROOT, "molmo2/scripts"))
from train_lora_vrsbench import VRS, load_vqa  # noqa: E402  (같은 분할·질문 정리)
from eval_vrsbench import PROMPTS  # noqa: E402

TARGET = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
IM_END = 151645
PAD = 151643


class VQADataset(Dataset):

    def __init__(self, rows, prompt, model_path):
        self.rows, self.prompt, self.model_path = rows, prompt, model_path
        self.processor = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.processor is None:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        r = self.rows[i]
        img = Image.open(join(VRS, "images/Images_train", r["image"])).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "text", "text": r["q"] + self.prompt},
                                             {"type": "image", "image": img}]}]
        x = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)
        prompt_ids = x["input_ids"][0]
        ans = torch.tensor(self.processor.tokenizer.encode(r["a"], add_special_tokens=False) + [IM_END],
                           dtype=prompt_ids.dtype)
        return dict(input_ids=torch.cat([prompt_ids, ans]),
                    labels=torch.cat([torch.full_like(prompt_ids, -100), ans]),
                    pixel_values=x["pixel_values"], image_grid_thw=x["image_grid_thw"])


def collate(items):
    L = max(len(it["input_ids"]) for it in items)
    B = len(items)
    batch = dict(input_ids=torch.full((B, L), PAD, dtype=items[0]["input_ids"].dtype),
                 labels=torch.full((B, L), -100, dtype=torch.long),
                 attention_mask=torch.zeros((B, L), dtype=torch.long))
    for b, it in enumerate(items):  # 오른쪽 padding (학습 전용, 손실은 labels로 제한)
        n = len(it["input_ids"])
        batch["input_ids"][b, :n] = it["input_ids"]
        batch["labels"][b, :n] = it["labels"]
        batch["attention_mask"][b, :n] = 1
    batch["pixel_values"] = torch.cat([it["pixel_values"] for it in items])
    batch["image_grid_thw"] = torch.cat([it["image_grid_thw"] for it in items])
    return batch


@torch.no_grad()
def evaluate(model, loader, max_batches=None):
    model.eval()
    tot, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        ntok = (batch["labels"][:, 1:] != -100).sum().item()
        tot += out.loss.item() * ntok
        n += ntok
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF 스냅샷 경로 (Qwen3-VL-4B/2B-Instruct)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="format_rules")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=0, help="0이면 epochs로 결정 (시험용 제한)")
    ap.add_argument("--val_frac", type=float, default=0.01)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val_batches", type=int, default=0, help="검증 배치 수 제한 (시험용)")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, get_cosine_schedule_with_warmup

    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "args.json", "w"), indent=1)
    log = open(out / "train_log.jsonl", "a")

    train_rows, val_rows = load_vqa(args.seed, args.val_frac)
    print(f"train {len(train_rows)} QA / val {len(val_rows)} QA, example: {train_rows[0]}", flush=True)
    prompt = PROMPTS[args.prompt]
    train_ds = VQADataset(train_rows, prompt, args.model)
    val_ds = VQADataset(val_rows, prompt, args.model)
    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, generator=g, collate_fn=collate,
                          num_workers=args.workers, pin_memory=True, persistent_workers=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, collate_fn=collate,
                        num_workers=args.workers // 2, pin_memory=True)

    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                             target_modules=TARGET, bias="none"))
    model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()
    tgt = sorted({n.split(".lora_A")[0].rsplit(".", 1)[-1] for n, _ in model.named_parameters() if "lora_A" in n})
    assert not any("visual" in n for n, p in model.named_parameters() if p.requires_grad), "시각부에 LoRA가 붙음"
    print("LoRA 대상 모듈 종류:", tgt, flush=True)

    steps_per_epoch = len(train_dl) // args.grad_accum
    total = args.max_steps or math.ceil(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, args.warmup, total)
    print(f"optimizer steps {total} (batch {args.bs} x accum {args.grad_accum})", flush=True)

    vl0 = evaluate(model, val_dl, max_batches=args.val_batches or (5 if args.max_steps else None))
    print(json.dumps(dict(step=0, val_loss=round(vl0, 4))), flush=True)
    log.write(json.dumps(dict(step=0, val_loss=round(vl0, 4))) + "\n")

    model.train()
    step, micro, t0, run_loss = 0, 0, time.time(), []
    done = False
    while not done:
        for batch in train_dl:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            run_loss.append(loss.item() * args.grad_accum)
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == 1:
                el = time.time() - t0
                rec = dict(step=step, loss=round(sum(run_loss) / len(run_loss), 4), lr=sched.get_last_lr()[0],
                           elapsed_s=round(el), eta_min=round(el / step * (total - step) / 60, 1),
                           mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                run_loss = []
            if step % args.eval_every == 0 or step == total:
                vl = evaluate(model, val_dl, max_batches=args.val_batches or None)
                rec = dict(step=step, val_loss=round(vl, 4), lr=sched.get_last_lr()[0])
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step % args.save_every == 0 or step == total:
                model.save_pretrained(out / ("adapter" if step == total else f"adapter_step{step}"))
            if step % steps_per_epoch == 0:
                ep = out / f"adapter_epoch{step // steps_per_epoch}"
                model.save_pretrained(ep)
                torch.save(dict(opt=opt.state_dict(), sched=sched.state_dict(), step=step), ep / "train_state.pt")
            if step >= total:
                done = True
                break
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
