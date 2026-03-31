# scripts/precompute_features.py
#
# Precompute and cache Qwen3-VL visual features for every unique video in
# splits/split_80_10_10 (train / val / test).
#
# Each .pt cache file contains:
#   visual_tokens      – [n_tokens, D_llm]        pooler output
#   deepstack_features – list of [n_tokens, D_llm] one per deepstack layer
#   video_grid_thw     – [1, 3]                    (T, H_grid, W_grid)
#   frames_indices     – list[int]                 frame indices in orig video
#   fps                – float                     original video FPS
#
# Single process (all videos):
#   python scripts/precompute_features.py --model Qwen/Qwen3-VL-2B-Instruct
#
# Specific shard (for SLURM array jobs):
#   python scripts/precompute_features.py --part 3 --total-parts 8 \
#       --model Qwen/Qwen3-VL-2B-Instruct

import os
import sys
import json
import argparse

import torch
from tqdm import tqdm

# Allow importing from scripts/ and models/
_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from vision_cache import video_feat_path, FEAT_DIR
from qwen_vl_utils import process_vision_info

from transformers import AutoProcessor
from models.qwen3_vl import Qwen3VLWithOfflineFeatures

SPLITS_DIR = "splits/split_80_10_10"
RESIZED_HEIGHT = 480
FPS = 1


# ---------------------------------------------------------------------------
# Video collection
# ---------------------------------------------------------------------------

def collect_videos() -> list[str]:
    """Return deduplicated list of all video paths across all splits."""
    seen: set[str] = set()
    paths: list[str] = []
    for split in ("train.json", "val.json", "test.json"):
        fpath = os.path.join(SPLITS_DIR, split)
        if not os.path.exists(fpath):
            print(f"[WARN] {fpath} not found, skipping.")
            continue
        with open(fpath) as f:
            data = json.load(f)
        for item in data:
            for vp in item["video"]:
                if vp not in seen:
                    seen.add(vp)
                    paths.append(vp)
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-2B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument("--part", type=int, default=0,
                        help="0-indexed shard index (for SLURM array jobs)")
    parser.add_argument("--total-parts", type=int, default=1,
                        help="Total number of shards")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.part >= args.total_parts:
        raise ValueError(f"--part ({args.part}) must be < --total-parts ({args.total_parts})")

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Load model and processor
    # ------------------------------------------------------------------
    print(f"Loading model {args.model!r} ...")
    model = Qwen3VLWithOfflineFeatures.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model)

    # ------------------------------------------------------------------
    # Shard selection
    # ------------------------------------------------------------------
    os.makedirs(FEAT_DIR, exist_ok=True)
    all_videos = collect_videos()
    video_paths = all_videos[args.part :: args.total_parts]

    print(
        f"Part {args.part + 1}/{args.total_parts} — "
        f"{len(video_paths)} / {len(all_videos)} videos"
    )

    skipped = 0
    errors: list[tuple[str, str]] = []

    for vp in tqdm(video_paths, desc=f"part {args.part}"):
        feat_path = video_feat_path(vp)
        if os.path.exists(feat_path):
            skipped += 1
            continue

        try:
            # ----------------------------------------------------------
            # 1. Build a single-video message and run qwen_vl_utils
            #    process_vision_info to get preprocessed frames + metadata.
            #    This mirrors the preprocessing done at eval time.
            # ----------------------------------------------------------
            messages = [{
                "role": "user",
                "content": [{
                    "type": "video",
                    "video": vp,
                    "resized_height": RESIZED_HEIGHT,
                    "fps": FPS,
                    "min_pixels": 16 * 28 * 28,
                    "max_pixels": 128 * 28 * 28,
                }],
            }]

            _, video_inputs, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            # video_inputs is a list of (tensor, metadata) tuples
            video_tensors, video_metadatas = zip(*video_inputs)
            video_tensors = list(video_tensors)
            video_metadatas = list(video_metadatas)

            vkw = dict(video_kwargs or {})
            vkw.pop("fps", None)  # avoid type conflict

            # ----------------------------------------------------------
            # 2. Run the video processor to get pixel patches.
            # ----------------------------------------------------------
            proc_out = processor(
                text=["<placeholder>"],
                videos=video_tensors,
                video_metadata=video_metadatas,
                return_tensors="pt",
                do_resize=False,
                **vkw,
            )

            pixel_values_videos = proc_out["pixel_values_videos"].to(device)
            video_grid_thw = proc_out["video_grid_thw"].to(device)  # [1, 3]

            # ----------------------------------------------------------
            # 3. Run the visual encoder in temporal chunks.
            # ----------------------------------------------------------
            visual_tokens, deepstack_features = model.encode_video(
                pixel_values_videos, video_grid_thw
            )
            # visual_tokens:      [n_tokens, D_llm]  on CPU (encode_video returns CPU)
            # deepstack_features: list of [n_tokens, D_llm] on CPU

            # ----------------------------------------------------------
            # 4. Extract metadata needed to reconstruct timestamps at eval.
            # ----------------------------------------------------------
            meta = video_metadatas[0]
            # metadata may be a dict or a dataclass depending on transformers version
            if isinstance(meta, dict):
                frames_indices = list(meta["frames_indices"])
                fps = float(meta["fps"])
            else:
                frames_indices = list(meta.frames_indices)
                fps = float(meta.fps)

            # ----------------------------------------------------------
            # 5. Save to cache.
            # ----------------------------------------------------------
            torch.save(
                {
                    "visual_tokens": visual_tokens,          # [n, D]
                    "deepstack_features": deepstack_features, # list[Tensor[n, D]]
                    "video_grid_thw": video_grid_thw.cpu(),  # [1, 3]
                    "frames_indices": frames_indices,
                    "fps": fps,
                },
                feat_path,
            )

        except Exception as e:
            errors.append((vp, str(e)))
            print(f"\n[ERROR] {vp}: {e}")

    done = len(video_paths) - skipped - len(errors)
    print(f"\nDone.  computed={done}  skipped(cached)={skipped}  errors={len(errors)}")
    if errors:
        print("Failed videos:")
        for vp, err in errors:
            print(f"  {vp}: {err}")


if __name__ == "__main__":
    main()
