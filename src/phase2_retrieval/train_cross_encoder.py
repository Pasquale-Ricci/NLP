"""
Phase 2b - Fine-tune a Cross-Encoder to rank KG triples against a claim.

  python train_cross_encoder.py --train data/rerank_train.jsonl --val data/rerank_val.jsonl \
      --base cross-encoder/ms-marco-MiniLM-L-6-v2 --out models/cross_ft --epochs 1

Then evaluate with:  python run_phase2.py ... --scorers lexical cross --cross-model models/cross_ft

Plain PyTorch + transformers loop (BCE on (claim, triple) pairs, AdamW, linear
warm-up/decay, fp16 autocast on GPU). The saved folder loads directly in
sentence-transformers' CrossEncoder. Validation reports, per claim, how often the
best-scored sampled triple is a positive (top1) and the pair accuracy.
"""

import argparse
import json
import math
import os
import random
from collections import defaultdict


def load_pairs(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def top1_rate(rows, scores):
    """Per claim: is the highest-scored pair a positive? -> fraction of claims."""
    best = {}
    for r, s in zip(rows, scores):
        c = r["claim"]
        if c not in best or s > best[c][0]:
            best[c] = (s, r["label"])
    return sum(lab for _, lab in best.values()) / max(len(best), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--base", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--out", default="models/cross_ft")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU detected (torch.cuda.is_available() is False): training on CPU "
              "will be very slow. Install the CUDA build of PyTorch.")
    use_amp = device == "cuda" and not args.no_fp16

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1).to(device)

    train, val = load_pairs(args.train), (load_pairs(args.val) if args.val else [])
    n_pos = sum(r["label"] for r in train)
    ratio = (len(train) - n_pos) / max(n_pos, 1)
    print(f"train pairs: {len(train)} (positives {n_pos}) | val pairs: {len(val)}")
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([math.sqrt(ratio)], device=device))   # tempered class weight

    steps = args.epochs * math.ceil(len(train) / args.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def encode(batch):
        return tok([r["claim"] for r in batch], [r["triple"] for r in batch], truncation=True,
                   max_length=args.max_len, padding=True, return_tensors="pt").to(device)

    @torch.no_grad()
    def score(rows):
        model.eval()
        out = []
        for i in range(0, len(rows), 128):
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                out += model(**encode(rows[i:i + 128])).logits.squeeze(-1).float().tolist()
        return out

    def validate():
        if not val:
            return
        s = score(val)
        acc = sum((x > 0) == (r["label"] == 1) for x, r in zip(s, val)) / len(val)
        print(f"  val: pair accuracy {acc:.3f} | top1-is-positive per claim {top1_rate(val, s):.3f}")

    validate()                       # baseline before training, for comparison
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)
        model.train()
        running, bar = 0.0, tqdm(range(0, len(train), args.bs), desc=f"epoch {epoch}")
        for step, i in enumerate(bar, 1):
            batch = train[i:i + args.bs]
            labels = torch.tensor([float(r["label"]) for r in batch], device=device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
                logits = model(**encode(batch)).logits.squeeze(-1)
            loss = loss_fn(logits.float(), labels)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 100 == 0:
                bar.set_postfix(loss=running / 100)
                running = 0.0
        validate()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "train_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()
