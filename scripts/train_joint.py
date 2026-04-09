# scripts/train_joint.py
#
# Joint training: LoRA fine-tuning of Qwen3-VL + ResolutionRouter.
#
# How the router is trained
# -------------------------
# The router predicts pool_level (0 / 1 / 2) from the user query.
# pool_level is a discrete choice, so lm_loss.backward() alone gives zero
# gradient to the router.  We fix this with one line:
#
#   router_loss = log_prob(chosen pool_level) * lm_loss.detach()
#
# This says: if this pool_level led to low LM loss, increase its probability.
# The router therefore learns to pick the resolution that helps the LM most.
#
# Gradient flow
# -------------
#   lm_loss      -> LoRA parameters   (language model improves)
#   router_loss  -> router parameters (router learns which level helps)

import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from dataset import VideoQADataset, collate_fn
from models.qwen3_vl import (
    Qwen3VLWithOfflineFeatures,
    Qwen3VLProcessorWithPrecomputed,
    extract_query_ids,
)
from models.resolution_router import ResolutionRouter
from models.resolution_pooling import prepare_pooled_inputs

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("[WARNING] peft not installed — training without LoRA")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
MODEL_NAME  = "Qwen/Qwen3-VL-2B-Instruct"
TRAIN_DATA  = "splits/split_70_15_15/train.json"
VAL_DATA    = "splits/split_70_15_15/val.json"
FEAT_DIR    = "/scratch/rai/vast1/alhalah/users/nikesh/qwen3vl_proj/features/qwen_video"
CACHE_DIR   = "cache/joint_dataset"
CKPT_DIR    = "checkpoints/joint"

EPOCHS             = 3
BATCH_SIZE         = 1       # keep 1 — each sample has a very long sequence
LR_LORA            = 2e-4
LR_ROUTER          = 1e-4
GRAD_CLIP          = 1.0
ROUTER_LOSS_WEIGHT = 0.1    # weight of router loss relative to LM loss

LORA_R       = 16
LORA_ALPHA   = 32
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]


# ---------------------------------------------------------------------------
# Label helper
# ---------------------------------------------------------------------------
def make_labels(seq_len: int, answer_tokens: torch.Tensor, device) -> torch.Tensor:
    """
    Build a [1, seq_len] label tensor for the LM loss.

    Only the answer tokens (at the very end of the sequence) are trained on.
    Everything else — padding, system prompt, question, video tokens — is
    set to -100 so the loss ignores them.

    This works because pooling only removes video tokens from the middle of
    the sequence, so the answer tokens always stay at the end.
    """
    n      = answer_tokens.shape[0]
    labels = torch.full((1, seq_len), -100, dtype=torch.long, device=device)
    labels[0, -n:] = answer_tokens.to(device)
    return labels


# ---------------------------------------------------------------------------
# Single forward step (shared by train and val)
# ---------------------------------------------------------------------------
def forward_step(sample, base_model, router, cfg, device, train: bool):
    """
    One forward pass through the router + pooling + language model.

    During training  (train=True):
        - Router samples pool_level stochastically (exploration).
        - Returns lm_loss + router_loss.

    During validation (train=False):
        - Router picks the most likely pool_level (greedy).
        - Returns lm_loss only.
    """
    input_ids = sample["input_ids"].to(device)
    attn_mask = sample["attention_mask"].to(device)
    feat_in   = [f.to(device) for f in sample["feature_inputs"]]
    vid_grid  = sample["video_grid_thw"]
    ds_in     = sample["deepstack_inputs"]
    if ds_in is not None:
        ds_in = [[t.to(device) for t in layer] for layer in ds_in]

    pad_id   = getattr(cfg, "pad_token_id", 0) or 0
    embed_fn = base_model.get_input_embeddings()

    # Step 1: extract the user's question tokens (text after the last video)
    query_ids = extract_query_ids(input_ids, cfg.vision_end_token_id, pad_id)

    # Step 2: router predicts a pool_level for this query
    logits = router(query_ids, embed_fn)   # [1, 3]

    if train:
        # Sample during training so the router explores all pool levels
        pool_level = int(torch.distributions.Categorical(logits=logits).sample().item())
    else:
        pool_level = int(logits.argmax(dim=-1).item())

    # Step 3: apply visual token pooling if needed
    if pool_level > 0:
        feat_in, vid_grid, input_ids, attn_mask, ds_in = prepare_pooled_inputs(
            feat_in, vid_grid, input_ids, attn_mask,
            pool_level=pool_level,
            video_token_id=cfg.video_token_id,
            vision_start_id=cfg.vision_start_token_id,
            vision_end_id=cfg.vision_end_token_id,
            deepstack_feature_inputs=ds_in,
            pad_token_id=pad_id,
        )

    # Step 4: run the language model
    labels = make_labels(input_ids.shape[1], sample["answer_tokens"], device)
    out    = base_model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=labels,
        feature_inputs=feat_in,
        deepstack_feature_inputs=ds_in,
        video_grid_thw=vid_grid,
    )
    lm_loss = out.loss

    result = {"lm_loss": lm_loss, "pool_level": pool_level}

    if train:
        # Step 5: router gradient
        # log_prob * lm_loss.detach() routes the LM signal back to the router:
        #   good choice (low lm_loss) -> push log_prob up
        #   bad  choice (high lm_loss) -> push log_prob down
        log_prob             = F.log_softmax(logits, dim=-1)[0, pool_level]
        result["router_loss"] = log_prob * lm_loss.detach()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load model ---
    print("Loading model ...")
    model = Qwen3VLWithOfflineFeatures.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto",
    )

    # --- apply LoRA ---
    if HAS_PEFT:
        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGETS,
            lora_dropout=0.05, bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        peft_model = get_peft_model(model, lora_cfg)
        peft_model.print_trainable_parameters()
        # Access the underlying model so our custom forward() is called
        base_model = peft_model.base_model.model
    else:
        peft_model = None
        base_model = model

    cfg       = base_model.config
    embed_fn  = base_model.get_input_embeddings()
    embed_dev = embed_fn.weight.device
    input_dim = embed_fn.weight.shape[1]  # 2048

    # --- processor ---
    processor    = Qwen3VLProcessorWithPrecomputed.from_pretrained(MODEL_NAME)
    pad_token_id = getattr(cfg, "pad_token_id", 0) or 0

    # --- datasets ---
    print("Building datasets ...")
    train_ds = VideoQADataset(
        TRAIN_DATA, FEAT_DIR, processor, pad_token_id,
        cache_path=os.path.join(CACHE_DIR, "train.pt"),
    )
    val_ds = VideoQADataset(
        VAL_DATA, FEAT_DIR, processor, pad_token_id,
        cache_path=os.path.join(CACHE_DIR, "val.pt"),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_fn, num_workers=0)

    # --- router ---
    router = ResolutionRouter(input_dim=input_dim).to(embed_dev)

    # --- optimizers (separate LR for LoRA vs router) ---
    lora_params = [p for p in base_model.parameters() if p.requires_grad]
    opt_lm  = torch.optim.AdamW(lora_params,         lr=LR_LORA,   weight_decay=1e-2)
    opt_rt  = torch.optim.AdamW(router.parameters(), lr=LR_ROUTER, weight_decay=1e-2)
    sched_lm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_lm, T_max=EPOCHS * len(train_loader))
    sched_rt = torch.optim.lr_scheduler.CosineAnnealingLR(opt_rt, T_max=EPOCHS * len(train_loader))

    best_val_loss = float("inf")

    # log file written alongside stdout (useful when running under slurm)
    log_path = os.path.join(CKPT_DIR, "train.log")
    log_file = open(log_path, "w", buffering=1)   # line-buffered

    def log(msg: str):
        print(msg)
        log_file.write(msg + "\n")

    log(f"{'epoch':>6} {'split':>6} {'lm_loss':>9} {'rt_loss':>9} "
        f"{'lr_lm':>9} {'lr_rt':>9} {'grad_norm':>10} {'levels':>14}")

    for epoch in range(1, EPOCHS + 1):

        # ---- training ----
        base_model.train()
        router.train()
        total_lm = total_rt = total_gnorm = 0.0
        level_counts = [0, 0, 0]
        t0 = time.time()

        bar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for step_idx, sample in enumerate(bar, 1):
            r = forward_step(sample, base_model, router, cfg, embed_dev, train=True)

            total_loss = r["lm_loss"] + ROUTER_LOSS_WEIGHT * r["router_loss"]

            opt_lm.zero_grad()
            opt_rt.zero_grad()
            total_loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(
                lora_params + list(router.parameters()), GRAD_CLIP
            ).item()
            opt_lm.step()
            opt_rt.step()
            sched_lm.step()
            sched_rt.step()

            lm_val = r["lm_loss"].item()
            rt_val = r["router_loss"].item()
            total_lm    += lm_val
            total_rt    += rt_val
            total_gnorm += gnorm
            level_counts[r["pool_level"]] += 1

            # live running averages in the tqdm bar
            bar.set_postfix(
                lm=f"{total_lm/step_idx:.4f}",
                rt=f"{total_rt/step_idx:.4f}",
                lvl=r["pool_level"],
                gnorm=f"{gnorm:.2f}",
            )

        n      = len(train_loader)
        lr_lm  = sched_lm.get_last_lr()[0]
        lr_rt  = sched_rt.get_last_lr()[0]
        elapsed = time.time() - t0

        log(f"{epoch:>6} {'train':>6} {total_lm/n:>9.4f} {total_rt/n:>9.4f} "
            f"{lr_lm:>9.2e} {lr_rt:>9.2e} {total_gnorm/n:>10.3f} "
            f"{str(level_counts):>14}  ({elapsed:.0f}s)")

        # ---- validation ----
        base_model.eval()
        router.eval()
        val_lm     = 0.0
        val_levels = [0, 0, 0]

        with torch.no_grad():
            for sample in tqdm(val_loader, desc=f"Epoch {epoch} [val]  "):
                r = forward_step(sample, base_model, router, cfg, embed_dev, train=False)
                val_lm += r["lm_loss"].item()
                val_levels[r["pool_level"]] += 1

        val_lm /= len(val_loader)
        log(f"{epoch:>6} {'val':>6} {val_lm:>9.4f} {'—':>9} "
            f"{'—':>9} {'—':>9} {'—':>10} {str(val_levels):>14}")

        # ---- save checkpoints ----
        epoch_dir = os.path.join(CKPT_DIR, f"epoch_{epoch:02d}")
        os.makedirs(epoch_dir, exist_ok=True)
        torch.save(router.state_dict(), os.path.join(epoch_dir, "router.pt"))
        if HAS_PEFT and peft_model:
            peft_model.save_pretrained(os.path.join(epoch_dir, "lora"))

        if val_lm < best_val_loss:
            best_val_loss = val_lm
            best_dir = os.path.join(CKPT_DIR, "best")
            os.makedirs(best_dir, exist_ok=True)
            torch.save(router.state_dict(), os.path.join(best_dir, "router.pt"))
            if HAS_PEFT and peft_model:
                peft_model.save_pretrained(os.path.join(best_dir, "lora"))
            print(f"  -> best checkpoint saved (val lm={val_lm:.4f})")

    log(f"\nTraining done.  Best val lm_loss: {best_val_loss:.4f}")
    log(f"  Router : {CKPT_DIR}/best/router.pt")
    if HAS_PEFT:
        log(f"  LoRA   : {CKPT_DIR}/best/lora/")
    log_file.close()


if __name__ == "__main__":
    main()
