# scripts/vision_cache.py
#
# Utilities for video feature caching.
#
# Two cache formats are supported:
#
#   Legacy  – raw pixel tensor [T, C, H, W] from fetch_video (used by
#             process_vision_info for the old eval path).
#
#   Current – dict produced by precompute_features.py:
#               {
#                 "visual_tokens":      Tensor[n, D],
#                 "deepstack_features": List[Tensor[n, D]],
#                 "video_grid_thw":     Tensor[1, 3],
#                 "frames_indices":     list[int],
#                 "fps":                float,
#               }
#             Load with load_precomputed_video().

import os
import hashlib
import torch
from typing import Optional, Union, List, Dict, Any


from qwen_vl_utils import extract_vision_info, fetch_image, fetch_video
from PIL import Image

FEAT_DIR="/scratch/rai/vast1/alhalah/users/nikesh/qwen3vl_proj/features/qwen_video"

# Use absolute path
FEAT_DIR = "/scratch/rai/vast1/alhalah/users/nikesh/qwen3vl_proj/features/qwen_video"


def video_feat_path(video_path: str, feat_dir: str = FEAT_DIR) -> str:
    """Derive a canonical cache path for a video's preprocessed tensor."""
    key = hashlib.md5(os.path.abspath(video_path).encode()).hexdigest()
    return os.path.join(feat_dir, f"{key}.pt")


def _load_or_compute_video(ele: Dict[str, Any]) -> torch.Tensor:
    """
    Return the preprocessed video tensor for *ele*.

    If ele contains "feat_path" and that file exists, load from disk.
    Otherwise run fetch_video and, if "feat_path" is set, save the result.
    """
    feat_path: Optional[str] = ele.get("feat_path")

    if feat_path and os.path.exists(feat_path):
        return torch.load(feat_path, weights_only=True)

    video_tensor, _sample_fps = fetch_video(ele, return_video_sample_fps=True)

    if feat_path:
        os.makedirs(os.path.dirname(feat_path), exist_ok=True)
        torch.save(video_tensor, feat_path)

    return video_tensor


def process_vision_info(
    conversations: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
    **kwargs,
):
    """
    Overloads qwen_vl_utils.process_vision_info.

    Identical contract: returns (image_inputs, video_inputs).
    Videos whose element dict carries a "feat_path" key are cached to / loaded
    from that path instead of being decoded every run.

    Any extra kwargs are silently ignored for drop-in compatibility.
    """
    vision_infos = extract_vision_info(conversations)
    image_inputs: List[Image.Image] = []
    video_inputs: List[torch.Tensor] = []

    for info in vision_infos:
        if "image" in info or "image_url" in info:
            image_inputs.append(fetch_image(info))
        elif "video" in info:
            video_inputs.append(_load_or_compute_video(info))
        else:
            raise ValueError("vision element must contain 'image', 'image_url', or 'video'")

    return image_inputs or None, video_inputs or None


def load_precomputed_video(feat_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a precomputed visual feature file (current format).

    Returns the dict saved by precompute_features.py:
        {
            "visual_tokens":      Tensor[n, D],
            "deepstack_features": List[Tensor[n, D]],
            "video_grid_thw":     Tensor[1, 3],
            "frames_indices":     list[int],
            "fps":                float,
        }
    Returns None if the file does not exist or is in the legacy pixel-tensor
    format.
    """
    if not os.path.exists(feat_path):
        print(f"[DEBUG vision_cache] MISS {feat_path}")
        return None
    data = torch.load(feat_path, weights_only=False)
    if isinstance(data, dict) and "visual_tokens" in data:
        vt = data["visual_tokens"]
        thw = data.get("video_grid_thw")
        print(f"[DEBUG vision_cache] HIT  {os.path.basename(feat_path)}  "
              f"visual_tokens={vt.shape}  grid_thw={thw.tolist() if thw is not None else None}  "
              f"fps={data.get('fps')}  frames={len(data.get('frames_indices', []))}")
        return data
    print(f"[DEBUG vision_cache] legacy format, skipping {feat_path}")
    return None  # legacy format
