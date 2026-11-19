"""Molmo2-4B LoRA 학습 (VRSBench train VQA) — 방법론 문서 E1 / E7 (c).

- LoRA: LLM의 attention(att_proj, attn_out)·MLP(ff_proj, ff_out)에만 적용, 시각부·connector 고정
- 기본은 causal(영상 블록 양방향 끔, token_type_ids 미전달). --bidirectional이면 원래 조건(양방향)으로 학습
- 입력 형식은 평가(eval_vrsbench.py)와 동일: 질문 + format_rules 지시문 + 이미지, add_generation_prompt=True
  그 뒤에 정답 + <|im_end|>를 직접 붙이고, 손실은 정답·<|im_end|> 토큰에만 건다
  (답변을 대화에 넣어 템플릿을 적용하면 평가 때와 다른 공백이 생기고 종료 토큰이 붙지 않아서)
- 검증: train 이미지의 1%를 떼어 검증 손실 계산 (test와 분리)

usage:
  HF_HOME=.../molmo2/hf_cache .../molmo2/.venv/bin/python train_lora_vrsbench.py --out <run_dir> [--max_steps 20]
"""
import argparse
import json
import math
import random
import re
import sys
import time
from os.path import join
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = "/home/user/ondeviceAI"
VRS = join(ROOT, "molmo2/data/VRSBench")
sys.path.insert(0, join(ROOT, "molmo2/scripts"))
from eval_vrsbench import PROMPTS  # noqa: E402

CKPT, REV = "allenai/Molmo2-4B", "042abfa7a38879a376cec03d949eff0aefaa0600"
TARGET = r".*transformer\.blocks\.\d+\.(self_attn\.(att_proj|attn_out)|mlp\.(ff_proj|ff_out))"
IM_END = 151645
PAD = 151643


# VRSBench train 질문은 LLaVA식 지시문 템플릿 9종으로 감싸져 있다. 평가 질문(test)은 순수 질문이므로
# 템플릿을 걷어내고, 평가와 같은 format_rules 지시문만 붙인다.
TEMPLATES = [
    r"^What is the answer to the following question\s+(?P<q>.+)$",
    r"^Based on the image, respond to this question with a short answer:\s*(?P<q>.+?)\.?$",
    r"^Use the provided image to answer the question:\s*(?P<q>.+?)\s*Provide your answer as short as possible\.$",
    r"^Given the image, answer the following question with no more than three words\.\s*(?P<q>.+)$",
    r"^Question:\s*(?P<q>.+?)\s*Short answer:$",
    r"^The question\s+(?P<q>.+?)\s+can be answered using the image\. A short answer is$",
    r"^Q:\s*(?P<q>.+?)\s*A:$",
    r"^(?P<q>.+?)\.?\s*A short answer to the question is$",
    r"^Question:\s*(?P<q>.+)$",
]


def clean_question(raw):
    q = re.sub(r"^<image>\s*\[vqa\]\s*", "", raw).strip()
    for t in TEMPLATES:
        m = re.match(t, q, flags=re.S)
        if m:
            return m.group("q").strip()
    return q


def load_vqa(seed, val_frac):
    data = json.load(open(join(VRS, "VRSBench_train.json")))
    rows = []
    for x in data:
        q = x["conversations"][0]["value"]
        if "[vqa]" not in q:
            continue
        rows.append(dict(image=x["image"], q=clean_question(q), a=x["conversations"][1]["value"].strip()))
    images = sorted({r["image"] for r in rows})
    random.Random(seed).shuffle(images)
    val_imgs = set(images[:max(1, int(len(images) * val_frac))])
    train = [r for r in rows if r["image"] not in val_imgs]
    val = [r for r in rows if r["image"] in val_imgs]
    return train, val


class VQADataset(Dataset):

    def __init__(self, rows, prompt, bidirectional):
        self.rows, self.prompt, self.bidirectional = rows, prompt, bidirectional
        self.processor = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self.processor is None:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(CKPT, revision=REV, trust_remote_code=True)
        r = self.rows[i]
        img = Image.open(join(VRS, "images/Images_train", r["image"])).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "text", "text": r["q"] + self.prompt},
                                             {"type": "image", "image": img}]}]
        x = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                               return_tensors="pt", return_dict=True)
        ans = self.processor.tokenizer.encode(r["a"], add_special_tokens=False) + [IM_END]
        prompt_ids = x["input_ids"][0]
        ans = torch.tensor(ans, dtype=prompt_ids.dtype)
        out = dict(
            input_ids=torch.cat([prompt_ids, ans]),
            labels=torch.cat([torch.full_like(prompt_ids, -100), ans]),
            pixel_values=x["pixel_values"],
            image_token_pooling=x["image_token_pooling"],
            image_grids=x["image_grids"],
            image_num_crops=x["image_num_crops"],
        )
        if self.bidirectional:
            out["token_type_ids"] = torch.cat([x["token_type_ids"][0], torch.zeros_like(ans, dtype=x["token_type_ids"].dtype)])
        return out


def collate(items):
    L = max(len(it["input_ids"]) for it in items)
    B = len(items)
    batch = dict(
        input_ids=torch.full((B, L), PAD, dtype=items[0]["input_ids"].dtype),
        labels=torch.full((B, L), -100, dtype=torch.long),
        attention_mask=torch.zeros((B, L), dtype=torch.long),
    )
    if "token_type_ids" in items[0]:
        batch["token_type_ids"] = torch.zeros((B, L), dtype=items[0]["token_type_ids"].dtype)
    for b, it in enumerate(items):
        n = len(it["input_ids"])
        batch["input_ids"][b, :n] = it["input_ids"]
        batch["labels"][b, :n] = it["labels"]
        batch["attention_mask"][b, :n] = 1
        if "token_type_ids" in it:
            batch["token_type_ids"][b, :n] = it["token_type_ids"]
    # 이미지 텐서는 이미지 단위로 이어 붙인다 (pooling 인덱스는 이미지별 로컬 인덱스라 그대로 연결)
    for k in ["pixel_values", "image_token_pooling", "image_grids", "image_num_crops"]:
        batch[k] = torch.cat([it[k] for it in items])
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


class PlateauLR:
    """검증 손실 개선 폭으로 학습률을 조절: warmup 후 유지, 개선이 없으면 절반으로 낮추고, 최소값에서도 없으면 학습 종료."""

    def __init__(self, opt, lr, warmup, factor, patience, threshold, min_lr):
        self.opt, self.warmup, self.factor, self.patience, self.threshold, self.min_lr = \
            opt, warmup, factor, patience, threshold, min_lr
        self.lr, self.steps, self.best, self.bad, self.stop = lr, 0, float("inf"), 0, False
        self._apply()

    def _apply(self):
        cur = self.lr * min(1.0, (self.steps + 1) / max(self.warmup, 1))
        for g in self.opt.param_groups:
            g["lr"] = cur

    def step(self):
        self.steps += 1
        self._apply()

    def get_last_lr(self):
        return [self.opt.param_groups[0]["lr"]]

    def report(self, val_loss):
        """개선되었으면 True를 돌려준다."""
        if val_loss < self.best * (1 - self.threshold):
            self.best, self.bad = val_loss, 0
            return True
        self.bad += 1
        if self.bad >= self.patience:
            self.bad = 0
            if self.lr <= self.min_lr:
                self.stop = True
            else:
                self.lr = max(self.lr * self.factor, self.min_lr)
                self._apply()
        return False

    def state_dict(self):
        return {k: getattr(self, k) for k in ["lr", "steps", "best", "bad", "stop"]}

    def load_state_dict(self, state):
        for k, v in state.items():
            setattr(self, k, v)
        self._apply()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--prompt", default="format_rules")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=0, help="0이면 epochs로 결정 (시험용 제한)")
    ap.add_argument("--val_frac", type=float, default=0.01)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--init_adapter", default=None, help="이 adapter에서 이어서 학습 (옵티마이저·학습률 일정은 새로 시작)")
    ap.add_argument("--base_epoch", type=int, default=0, help="이미 학습한 epoch 수. epoch 저장 폴더 번호에만 사용")
    ap.add_argument("--shuffle_seed", type=int, default=None, help="학습 순서 seed. 검증 분리 seed와 별개")
    ap.add_argument("--sched", default="cosine", choices=["cosine", "plateau"])
    ap.add_argument("--plateau_factor", type=float, default=0.5)
    ap.add_argument("--plateau_patience", type=int, default=2, help="개선 없는 검증 횟수가 이만큼 쌓이면 학습률을 낮춤")
    ap.add_argument("--plateau_threshold", type=float, default=0.005, help="이 상대 비율 이상 줄어야 개선으로 침")
    ap.add_argument("--min_lr", type=float, default=1.25e-5)
    ap.add_argument("--epoch_steps", type=int, default=0, help="epoch 저장 간격 (시험용)")
    ap.add_argument("--val_batches", type=int, default=0, help="검증 배치 수 제한 (시험용)")
    ap.add_argument("--resume", default=None, help="adapter_epochN 폴더. 가중치·옵티마이저·학습률 일정까지 복원")
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
    train_ds = VQADataset(train_rows, prompt, args.bidirectional)
    val_ds = VQADataset(val_rows, prompt, args.bidirectional)
    g = torch.Generator().manual_seed(args.seed if args.shuffle_seed is None else args.shuffle_seed)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True, generator=g, collate_fn=collate,
                          num_workers=args.workers, pin_memory=True, persistent_workers=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.bs, shuffle=False, collate_fn=collate,
                        num_workers=args.workers // 2, pin_memory=True)

    model = AutoModelForImageTextToText.from_pretrained(CKPT, revision=REV, trust_remote_code=True,
                                                        dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    init = args.resume or args.init_adapter
    if init:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, init, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                                 target_modules=TARGET, bias="none"))
    # PEFT 감싼 뒤에 켠다: 먼저 켜 두면 PEFT가 입력 임베딩에 requires_grad hook을 달고,
    # Molmo2의 영상 특징 in-place 덧셈(x[is_image_patch] += ...)과 충돌한다. non-reentrant라 hook 불필요.
    model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    steps_per_epoch = args.epoch_steps or len(train_dl) // args.grad_accum
    total = args.max_steps or math.ceil(steps_per_epoch * args.epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    if args.sched == "plateau":
        sched = PlateauLR(opt, args.lr, args.warmup, args.plateau_factor, args.plateau_patience,
                          args.plateau_threshold, args.min_lr)
    else:
        sched = get_cosine_schedule_with_warmup(opt, args.warmup, total)
    start_step = 0
    if args.resume:
        state = torch.load(join(args.resume, "train_state.pt"), map_location="cuda")
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_step = state["step"]
        print(f"resume from step {start_step}", flush=True)
    print(f"optimizer steps {total} (batch {args.bs} x accum {args.grad_accum}), mode "
          f"{'bidirectional' if args.bidirectional else 'causal'}", flush=True)

    vl0 = evaluate(model, val_dl, max_batches=args.val_batches or (5 if args.max_steps else None))
    print(json.dumps(dict(step=0, val_loss=round(vl0, 4))), flush=True)
    log.write(json.dumps(dict(step=0, val_loss=round(vl0, 4))) + "\n")

    model.train()
    step, micro, t0, run_loss = start_step, 0, time.time(), []
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
                           elapsed_s=round(el), eta_min=round(el / (step - start_step) * (total - step) / 60, 1),
                           mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                run_loss = []
            if step % args.eval_every == 0 or step == total:
                vl = evaluate(model, val_dl, max_batches=args.val_batches or None)
                rec = dict(step=step, val_loss=round(vl, 4), lr=sched.get_last_lr()[0])
                if args.sched == "plateau":
                    if sched.report(vl):
                        model.save_pretrained(out / "adapter_best")
                        rec["best"] = True
                    rec["lr_next"] = sched.lr
                    rec["stop"] = sched.stop
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            stop = getattr(sched, "stop", False)
            if step % args.save_every == 0 or step == total or stop:
                model.save_pretrained(out / ("adapter" if step == total or stop else f"adapter_step{step}"))
            if step % steps_per_epoch == 0:
                ep = out / f"adapter_epoch{args.base_epoch + step // steps_per_epoch}"
                model.save_pretrained(ep)
                torch.save(dict(opt=opt.state_dict(), sched=sched.state_dict(), step=step), ep / "train_state.pt")
            if step >= total or stop:
                done = True
                break
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
