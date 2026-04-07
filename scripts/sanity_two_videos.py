# scripts/sanity_two_videos.py
#
# Sanity check: load two videos, ask the model to describe each and compare them.
# Uses precomputed features if available, otherwise falls back to online decoding.

import os
import sys
import torch
import argparse


def _to_device(v, device):
    if isinstance(v, torch.Tensor):
        return v.to(device)
    if isinstance(v, list):
        return [_to_device(x, device) for x in v]
    return v

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from vision_cache import video_feat_path, load_precomputed_video
from qwen_vl_utils import process_vision_info
from models.qwen3_vl import Qwen3VLWithOfflineFeatures, Qwen3VLProcessorWithPrecomputed

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
RESIZED_HEIGHT = 480
FPS = 1
MAX_NEW_TOKENS = 512

DEFAULT_VIDEO_1 = "/scratch/general/vast/u1209255/CrossVid_Dataset/videos/cook/iuQjb1-WAzs.mp4"
DEFAULT_VIDEO_2 = "/scratch/general/vast/u1209255/CrossVid_Dataset/videos/movie/68.mp4"
PROMPT = (
    "Please describe what is happening in Video 1 and Video 2 separately, "
    "then explain the main differences between them."
)

    
def build_messages(video_path1, video_path2, prompt):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "This is Video 1."},
                {
                    "type": "video",
                    "video": video_path1,
                    "resized_height": RESIZED_HEIGHT,
                    "fps": FPS,
                    "min_pixels": 16 * 28 * 28,
                    "max_pixels": 128 * 28 * 28,
                },
                {"type": "text", "text": "This is Video 2."},
                {
                    "type": "video",
                    "video": video_path2,
                    "resized_height": RESIZED_HEIGHT,
                    "fps": FPS,
                    "min_pixels": 16 * 28 * 28,
                    "max_pixels": 128 * 28 * 28,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def load_feat(video_path):
    return load_precomputed_video(video_feat_path(video_path))


def run_offline(video_path1, video_path2, prompt, processor):
    feat1 = load_feat(video_path1)
    feat2 = load_feat(video_path2)
    if feat1 is None or feat2 is None:
        missing = []
        if feat1 is None:
            missing.append(video_path1)
        if feat2 is None:
            missing.append(video_path2)
        print(f"[INFO] Missing precomputed features for: {missing}")
        return None

    messages = build_messages(video_path1, video_path2, prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    precomputed_info = []
    all_visual_tokens = []
    all_grid_thw = []
    all_deepstack = None

    for feat in (feat1, feat2):
        precomputed_info.append({
            "video_grid_thw": feat["video_grid_thw"].squeeze(0),
            "frames_indices": feat["frames_indices"],
            "fps": feat["fps"],
        })
        all_visual_tokens.append(feat["visual_tokens"])
        all_grid_thw.append(feat["video_grid_thw"])

        ds = feat.get("deepstack_features") or []
        if ds:
            if all_deepstack is None:
                all_deepstack = [[] for _ in range(len(ds))]
            for li, d in enumerate(ds):
                all_deepstack[li].append(d)

    proc_out = processor.call_with_precomputed(
        [text],
        precomputed_info=precomputed_info,
        padding=True,
        return_tensors="pt",
    )

    video_grid_thw = all_grid_thw                # List[Tensor[1,3]]
    feature_inputs = all_visual_tokens          # List[Tensor[n_i, D]]
    deepstack_feature_inputs = all_deepstack    # List[List[Tensor]] or None

    inputs = {
        **proc_out,
        "video_grid_thw": video_grid_thw,
        "feature_inputs": feature_inputs,
        "deepstack_feature_inputs": deepstack_feature_inputs,
    }
    return inputs


def run_online(video_path1, video_path2, prompt, processor):
    messages = build_messages(video_path1, video_path2, prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    videos, video_metadatas = [], []
    if video_inputs:
        for tensor, meta in video_inputs:
            videos.append(tensor)
            video_metadatas.append(meta)

    vkw = dict(video_kwargs or {})
    vkw.pop("fps", None)

    proc_out = processor(
        text=[text],
        images=image_inputs or None,
        videos=videos or None,
        video_metadata=video_metadatas or None,
        padding=True,
        return_tensors="pt",
        do_resize=False,
        return_mm_token_type_ids=True,
        **vkw,
    )
    return proc_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video1", default=DEFAULT_VIDEO_1, help="Path to first video")
    parser.add_argument("--video2", default=DEFAULT_VIDEO_2, help="Path to second video")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    print(f"Video 1: {args.video1}")
    print(f"Video 2: {args.video2}")
    print(f"Prompt : {args.prompt}")
    print()

    print("Loading model...")
    model = Qwen3VLWithOfflineFeatures.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()

    processor = Qwen3VLProcessorWithPrecomputed.from_pretrained(args.model)
    processor.tokenizer.padding_side = "left"

    # Try offline first
    inputs = run_offline(args.video1, args.video2, args.prompt, processor)
    if inputs is not None:
        print("[INFO] Using precomputed features (offline mode)")
    else:
        print("[ERROR] Precomputed features not found for one or both videos.")
        exit()
        

    inputs = {k: _to_device(v, model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    generated_ids_trimmed = generated_ids[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated_ids_trimmed, skip_special_tokens=True)

    print("=" * 60)
    print("Model response:")
    print("=" * 60)
    print(response)


if __name__ == "__main__":
    main()
