# scripts/train_joint.py
#
# Joint training: LoRA fine-tuning of Qwen3-VL + ResolutionRouter.
# Uses HuggingFace Accelerate with device_map pipeline parallelism.
# The model is split across all 4 GPUs via device_map="auto" (single process).
# Layers are assigned in order so the lm_head lives on GPU 3, keeping the
# ~24 GB logits tensor on the GPU with the most headroom.
#
# Why not ZeRO-3 + DDP
# ----------------------
# ZeRO-3 shards parameters but NOT activations. Each of the 4 ranks still
# materialises the full logits tensor (seq_len × 151936 × 2 bytes) for its own
# sample — at 86K tokens that is 24 GB, which exceeds free memory on a rank
# that already holds gathered params.  With device_map, only GPU 3 ever sees
# the logits, and it has ~78 GB free for them.
#
# How the router is trained
# -------------------------
# The router predicts pool_level (0 / 1 / 2) from the user query.
# pool_level is a discrete choice, so lm_loss.backward() alone gives zero
# gradient to the router.  We fix this with one line:
#
#   router_loss = log_prob(chosen pool_level) * lm_loss.detach()
#
# Gradient flow
# -------------
#   lm_loss      -> LoRA parameters   (language model improves)
#   router_loss  -> router parameters (router learns which level helps)
#
# Launching
# ---------
#   accelerate launch --num_processes 1 --mixed_precision bf16 scripts/train_joint.py

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerate import Accelerator
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_vl

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("[WARNING] wandb not installed — logging disabled")

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
CACHE_DIR   = "/scratch/rai/vast1/alhalah/users/nikesh/qwen3vl_proj/cache/joint_dataset"
CKPT_DIR    = "checkpoints/joint"

EPOCHS             = 3
BATCH_SIZE         = 1
LR_LORA            = 2e-4
LR_ROUTER          = 1e-4
GRAD_CLIP          = 1.0
ROUTER_LOSS_WEIGHT = 0.1

LORA_R       = 16
LORA_ALPHA   = 32
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ---------------------------------------------------------------------------
# Label helper
# ---------------------------------------------------------------------------
def make_labels(seq_len: int, answer_tokens: torch.Tensor, device) -> torch.Tensor:
    """
    Build a [1, seq_len] label tensor for the LM loss.
    Only answer tokens (at the tail of the sequence) are trained on;
    everything else is -100 so the loss ignores it.
    """
    n      = answer_tokens.shape[0]
    labels = torch.full((1, seq_len), -100, dtype=torch.long, device=device)
    labels[0, -n:] = answer_tokens.to(device)
    return labels


# ---------------------------------------------------------------------------
# Single forward step (shared by train and val)
# ---------------------------------------------------------------------------
def forward_step(sample, model, embed_fn, router, cfg, device, train: bool):
    """
    One forward pass through the router + pooling + language model.

    model    — peft_model with device_map spreading layers across all GPUs.
    embed_fn — embedding layer (on GPU 0); router uses it read-only.

    During training  (train=True):  router samples stochastically.
    During validation (train=False): router picks argmax.
    """
    input_ids = sample["input_ids"].to(device)
    attn_mask = sample["attention_mask"].to(device)
    feat_in   = [f.to(device) for f in sample["feature_inputs"]]
    vid_grid  = sample["video_grid_thw"]
    # ds_in     = sample["deepstack_inputs"]
    # if ds_in is not None:
    #     ds_in = [[t.to(device) for t in layer] for layer in ds_in]

    pad_id = getattr(cfg, "pad_token_id", 0) or 0

    # Step 1: extract query tokens for the router
    query_ids = extract_query_ids(input_ids, cfg.vision_end_token_id, pad_id)

    # Step 2: router predicts pool_level
    logits = router(query_ids, embed_fn)   # [1, 3]

    if train:
        pool_level = int(torch.distributions.Categorical(logits=logits).sample().item())
    else:
        pool_level = int(logits.argmax(dim=-1).item())

    # Step 3: apply visual token pooling if needed
    if pool_level > 0:
        feat_in, vid_grid, input_ids, attn_mask = prepare_pooled_inputs(
            feat_in, vid_grid, input_ids, attn_mask,
            pool_level=pool_level,
            video_token_id=cfg.video_token_id,
            vision_start_id=cfg.vision_start_token_id,
            vision_end_id=cfg.vision_end_token_id,
            pad_token_id=pad_id,
        )

    # Step 4: run the language model
    labels = make_labels(input_ids.shape[1], sample["answer_tokens"], device)
    out = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=labels,
        feature_inputs=feat_in,
        deepstack_feature_inputs=None,
        video_grid_thw=vid_grid,
    )
    lm_loss = out.loss

    result = {"lm_loss": lm_loss, "labels": labels, "pool_level": pool_level}

    if train:
        log_prob = F.log_softmax(logits, dim=-1)[0, pool_level]
        # lm_loss is on the last GPU; move to router device (GPU 0) for the scalar multiply
        result["router_loss"] = log_prob * lm_loss.detach().to(log_prob.device)

    return result



# ---------------------------------------------------------------------------
# Checkpoint helper
# ---------------------------------------------------------------------------
def save_checkpoint(accelerator: Accelerator, peft_model, router, save_dir: str):
    """Save LoRA adapter weights + router. Single-process, so write directly."""
    os.makedirs(save_dir, exist_ok=True)
    torch.save(router.state_dict(), os.path.join(save_dir, "router.pt"))
    if HAS_PEFT and peft_model is not None:
        accelerator.unwrap_model(peft_model).save_pretrained(os.path.join(save_dir, "lora"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat-dir",    default=FEAT_DIR)
    parser.add_argument("--cache-dir",   default=CACHE_DIR)
    parser.add_argument("--ckpt-dir",    default=CKPT_DIR)
    parser.add_argument("--wandb-project", default="vlm_router")
    parser.add_argument("--wandb-run",     default=None)
    args = parser.parse_args()

    feat_dir  = args.feat_dir
    cache_dir = args.cache_dir
    ckpt_dir  = args.ckpt_dir

    # --- Single-process Accelerate + device_map pipeline parallelism ---
    # The model is loaded with device_map="auto" which assigns transformer layers
    # across all visible GPUs.  One process drives all GPUs; no DDP, no DeepSpeed.
    accelerator = Accelerator(mixed_precision="bf16")
    is_main = accelerator.is_main_process

    os.makedirs(ckpt_dir, exist_ok=True)

    # --- wandb ---
    if is_main and HAS_WANDB:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=dict(
                model=MODEL_NAME,
                epochs=EPOCHS,
                lr_lora=LR_LORA,
                lr_router=LR_ROUTER,
                lora_r=LORA_R,
                lora_alpha=LORA_ALPHA,
                grad_clip=GRAD_CLIP,
                router_loss_weight=ROUTER_LOSS_WEIGHT,
                feat_dir=feat_dir,
            ),
        )

    # Fuse lm_head + cross-entropy to avoid materialising the full
    # [seq_len × 151936] logits tensor (would OOM on long sequences).
    # All other liger kernels (rms_norm, rope, swiglu) are left off —
    # LigerRMSNorm produces NaN in bf16 on long sequences.
    apply_liger_kernel_to_qwen3_vl(
        rope=False,
        rms_norm=False,
        swiglu=False,
        cross_entropy=False,
        fused_linear_cross_entropy=True,
    )

    # --- load model ---
    if is_main:
        print("Loading model ...")
    model = Qwen3VLWithOfflineFeatures.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",   # spread layers across all visible GPUs
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
        peft_model = peft_model.to(torch.bfloat16)
        if is_main:
            peft_model.print_trainable_parameters()
        # Gradient checkpointing: recompute activations during backward instead of
        # storing them, trading compute for activation memory on long sequences.
        peft_model.enable_input_require_grads()
        peft_model.gradient_checkpointing_enable()
        base_model = peft_model.base_model.model
        train_model = peft_model
    else:
        peft_model  = None
        base_model  = model
        train_model = model

    cfg = base_model.config
    embed_fn  = base_model.get_input_embeddings()
    input_dim = embed_fn.embedding_dim
    # With device_map the embedding lives on GPU 0; use that as the input device.
    device = embed_fn.weight.device

    # --- processor ---
    processor    = Qwen3VLProcessorWithPrecomputed.from_pretrained(MODEL_NAME)
    pad_token_id = getattr(cfg, "pad_token_id", 0) or 0

    # --- datasets ---
    if is_main:
        print("Building datasets ...")
    train_ds = VideoQADataset(
        TRAIN_DATA, feat_dir, processor, pad_token_id,
        cache_path=os.path.join(cache_dir, "train.pt"),
    )
    val_ds = VideoQADataset(
        VAL_DATA, feat_dir, processor, pad_token_id,
        cache_path=os.path.join(cache_dir, "val.pt"),
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # --- router — on GPU 0 alongside the embedding layer ---
    router = ResolutionRouter(input_dim=input_dim).to(device)

    # --- optimizers ---
    lora_params = [p for p in train_model.parameters() if p.requires_grad]
    opt_lm  = torch.optim.AdamW(lora_params,         lr=LR_LORA,   weight_decay=1e-2)
    opt_rt  = torch.optim.AdamW(router.parameters(), lr=LR_ROUTER, weight_decay=1e-2)

    # device_map handles model placement; only prepare dataloaders + optimizer
    opt_lm, train_loader, val_loader = accelerator.prepare(
        opt_lm, train_loader, val_loader,
    )

    sched_lm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_lm, T_max=EPOCHS * len(train_loader))
    sched_rt = torch.optim.lr_scheduler.CosineAnnealingLR(opt_rt, T_max=EPOCHS * len(train_loader))

    best_val_loss = float("inf")

    log_path = os.path.join(ckpt_dir, "train.log")
    log_file = open(log_path, "w", buffering=1) if is_main else None

    def log(msg: str):
        if is_main:
            print(msg)
            log_file.write(msg + "\n")

    log(f"{'epoch':>6} {'split':>6} {'lm_loss':>9} {'rt_loss':>9} "
        f"{'lr_lm':>9} {'lr_rt':>9} {'grad_norm':>10} {'levels':>14}")

    for epoch in range(1, EPOCHS + 1):

        # ---- training ----
        train_model.train()
        router.train()
        total_lm = total_rt = total_gnorm = 0.0
        level_counts = [0, 0, 0]
        t0 = time.time()

        bar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", disable=not is_main)
        for step_idx, sample in enumerate(bar, 1):
            r = forward_step(sample, train_model, embed_fn, router, cfg, device, train=True)

            lm_loss = r["lm_loss"]

            # router_loss is on GPU 0; lm_loss is on last GPU — align before summing
            total_loss = lm_loss + ROUTER_LOSS_WEIGHT * r["router_loss"].to(lm_loss.device)

            opt_lm.zero_grad()
            opt_rt.zero_grad()
            accelerator.backward(total_loss)
            gnorm = float(torch.nn.utils.clip_grad_norm_(train_model.parameters(), GRAD_CLIP))
            torch.nn.utils.clip_grad_norm_(router.parameters(), GRAD_CLIP)
            opt_lm.step()
            opt_rt.step()
            sched_lm.step()
            sched_rt.step()

            total_lm    += lm_loss.item()
            total_rt    += r["router_loss"].item()
            total_gnorm += gnorm
            level_counts[r["pool_level"]] += 1

            if is_main and HAS_WANDB:
                global_step = (epoch - 1) * len(train_loader) + step_idx
                wandb.log({
                    "train/lm_loss":    lm_loss.item(),
                    "train/router_loss": r["router_loss"].item(),
                    "train/grad_norm":  gnorm,
                    "train/pool_level": r["pool_level"],
                    "train/lr_lm":      sched_lm.get_last_lr()[0],
                    "train/lr_router":  sched_rt.get_last_lr()[0],
                }, step=global_step)

            bar.set_postfix(
                lm=f"{total_lm/step_idx:.4f}",
                rt=f"{total_rt/step_idx:.4f}",
                lvl=r["pool_level"],
                gnorm=f"{gnorm:.2f}",
            )

        n       = len(train_loader)
        lr_lm   = sched_lm.get_last_lr()[0]
        lr_rt   = sched_rt.get_last_lr()[0]
        elapsed = time.time() - t0

        log(f"{epoch:>6} {'train':>6} {total_lm/n:>9.4f} {total_rt/n:>9.4f} "
            f"{lr_lm:>9.2e} {lr_rt:>9.2e} {total_gnorm/n:>10.3f} "
            f"{str(level_counts):>14}  ({elapsed:.0f}s)")

        if is_main and HAS_WANDB:
            wandb.log({
                "epoch/train_lm_loss":    total_lm / n,
                "epoch/train_router_loss": total_rt / n,
                "epoch/train_grad_norm":  total_gnorm / n,
                "epoch/level_0": level_counts[0],
                "epoch/level_1": level_counts[1],
                "epoch/level_2": level_counts[2],
            }, step=epoch * len(train_loader))

        # ---- validation ----
        train_model.eval()
        router.eval()
        val_lm     = 0.0
        val_levels = [0, 0, 0]

        with torch.no_grad():
            for sample in tqdm(val_loader, desc=f"Epoch {epoch} [val]  ", disable=not is_main):
                r = forward_step(sample, train_model, embed_fn, router, cfg, device, train=False)
                lm_loss = r["lm_loss"]
                val_lm += lm_loss.item()
                val_levels[r["pool_level"]] += 1

        val_lm /= len(val_loader)
        log(f"{epoch:>6} {'val':>6} {val_lm:>9.4f} {'—':>9} "
            f"{'—':>9} {'—':>9} {'—':>10} {str(val_levels):>14}")

        if is_main and HAS_WANDB:
            wandb.log({
                "epoch/val_lm_loss": val_lm,
                "epoch/val_level_0": val_levels[0],
                "epoch/val_level_1": val_levels[1],
                "epoch/val_level_2": val_levels[2],
            }, step=epoch * len(train_loader))

        # ---- checkpoints ----
        epoch_dir = os.path.join(ckpt_dir, f"epoch_{epoch:02d}")
        save_checkpoint(accelerator, peft_model, router, epoch_dir)

        if val_lm < best_val_loss:
            best_val_loss = val_lm
            best_dir = os.path.join(ckpt_dir, "best")
            save_checkpoint(accelerator, peft_model, router, best_dir)
            if is_main:
                print(f"  -> best checkpoint saved (val lm={val_lm:.4f})")

    log(f"\nTraining done.  Best val lm_loss: {best_val_loss:.4f}")
    log(f"  Router : {ckpt_dir}/best/router.pt")
    if HAS_PEFT:
        log(f"  LoRA   : {ckpt_dir}/best/lora/")
    if log_file:
        log_file.close()
    if is_main and HAS_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
