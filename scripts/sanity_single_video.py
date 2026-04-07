# scripts/sanity_single_video.py
#
# Sanity check: load one video and ask the model to describe it.
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
MAX_NEW_TOKENS = 256
FEAT_DIR = os.path.join(_HERE, "..", "features", "qwen_video")

DEFAULT_VIDEO = "/scratch/general/vast/u1209255/CrossVid_Dataset/videos/cook/qRUbmEZ6oiA.mp4"
PROMPT = "Please describe in detail what is happening in this video."


def build_messages(video_path, prompt):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "resized_height": RESIZED_HEIGHT,
                    "fps": FPS,
                    "min_pixels": 16 * 28 * 28,
                    "max_pixels": 128 * 28 * 28,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def run_offline(video_path, prompt, processor):
    feat_path = video_feat_path(video_path, feat_dir=FEAT_DIR)
    feat = load_precomputed_video(feat_path)
    if feat is None:
        return None

    messages = build_messages(video_path, prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    precomputed_info = [{
        "video_grid_thw": feat["video_grid_thw"].squeeze(0),
        "frames_indices": feat["frames_indices"],
        "fps": feat["fps"],
    }]

    proc_out = processor.call_with_precomputed(
        [text],
        precomputed_info=precomputed_info,
        padding=True,
        return_tensors="pt",
    )

    feature_inputs = [feat["visual_tokens"]]        # List[Tensor[n, D]]
    video_grid_thw = [feat["video_grid_thw"]]       # List[Tensor[1,3]]

    ds = feat.get("deepstack_features") or []
    # deepstack_feature_inputs: List[List[Tensor]] — outer=layer, inner=video
    deepstack_feature_inputs = [[d] for d in ds] if ds else None

    inputs = {
        **proc_out,
        "video_grid_thw": video_grid_thw,
        "feature_inputs": feature_inputs,
        "deepstack_feature_inputs": deepstack_feature_inputs,
    }
    return inputs


def run_online(video_path, prompt, processor):
    messages = build_messages(video_path, prompt)
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
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="Path to video file")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    print(f"Video : {args.video}")
    print(f"Prompt: {args.prompt}")
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
    inputs = run_offline(args.video, args.prompt, processor)
    if inputs is not None:
        print("[INFO] Using precomputed features (offline mode)")
    else:
        print("[INFO] No precomputed features found — decoding frames on-the-fly (online mode)")
        exit()
        # inputs = run_online(args.video, args.prompt, processor)

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
