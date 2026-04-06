# scripts/eval_qwen_video.py
#
# Evaluate Qwen3-VL on the video QA test split.
#
# When precomputed visual features are present (from precompute_features.py)
# the visual encoder is skipped entirely; otherwise the model processes video
# frames on-the-fly (slower but correct).

import os
import re
import sys
import json
import torch
from tqdm import tqdm

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from vision_cache import video_feat_path, load_precomputed_video
from qwen_vl_utils import process_vision_info

from models.qwen3_vl import Qwen3VLWithOfflineFeatures, Qwen3VLProcessorWithPrecomputed

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DATA_PATH = "splits/split_70_15_15/test.json"
OUT_PATH = "results/qwen_video_eval.jsonl"
RESIZED_HEIGHT = 480
FPS = 1
MAX_NEW_TOKENS = 96
BATCH_SIZE = 1
LIMIT = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_task(sample_id):
    return sample_id.split("_")[0]


def extract_pred(response, task):
    """Extract the model prediction from a raw response string, task-aware."""
    text = response.strip()
    if task in ("CC", "NC", "PEA", "PI"):
        # Single or multi-select letter (A-F)
        letters = list(dict.fromkeys(re.findall(r"\b([A-F])\b", text.upper())))
        return ",".join(sorted(letters)) if letters else None
    elif task == "FSA":
        # Time interval: expect "start,end" — extract first two numbers
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        return f"{nums[0]},{nums[1]}" if len(nums) >= 2 else None
    elif task == "PSS":
        # Ordering: "2->3->1->4"
        m = re.search(r"\d+(?:->\d+)+", text)
        return m.group(0) if m else None
    return None


def normalize_gt(gt, task):
    """Normalize ground truth for comparison."""
    if task in ("CC", "NC", "PEA", "PI"):
        letters = re.findall(r"[A-F]", gt.strip().upper())
        return ",".join(sorted(letters)) if letters else gt.strip()
    return gt.strip()


def temporal_iou(pred_str, gt_str):
    """IoU between two time segments given as 'start,end' strings."""
    try:
        ps, pe = map(float, pred_str.split(","))
        gs, ge = map(float, gt_str.split(","))
    except Exception:
        return 0.0
    inter = max(0.0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def build_messages(video_paths, prompt):
    content = []
    for i, vp in enumerate(video_paths):
        content.append({
            "type": "video",
            "video": vp,
            "resized_height": RESIZED_HEIGHT,
            "fps": FPS,
            "min_pixels": 16 * 28 * 28,
            "max_pixels": 128 * 28 * 28,
        })
        content.append({"type": "text", "text": f"This is Video {i + 1}."})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def load_precomputed_for_video(vp):
    """Load precomputed dict for one video, or return None."""
    return load_precomputed_video(video_feat_path(vp))


# ---------------------------------------------------------------------------
# Batch assembly helpers
# ---------------------------------------------------------------------------

def assemble_offline_batch(batch, processor):
    """
    Build model inputs for a batch using precomputed visual features.

    Returns (inputs_dict, batch_meta) where inputs_dict is ready for
    model.generate(), or raises RuntimeError if any video is missing features.
    """
    texts = []
    all_precomputed_info = []  # one entry per video across the whole batch
    all_visual_tokens = []
    all_deepstack = None       # list of per-layer lists, built lazily
    all_grid_thw = []
    batch_meta = []

    for item in batch:
        sample_id = item["id"]
        video_paths = item["video"]
        prompt = item["conversations"][0]["value"]
        gt = item["conversations"][1]["value"].strip().upper()

        # Load precomputed features for all videos in this sample
        if not video_paths:
            raise RuntimeError(f"Sample {sample_id} has no video paths")
        sample_precomputed = []
        for vp in video_paths:
            feat = load_precomputed_for_video(vp)
            if feat is None:
                raise RuntimeError(f"Missing precomputed features for {vp}")
            sample_precomputed.append(feat)

        messages = build_messages(video_paths, prompt)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        texts.append(text)
        batch_meta.append((sample_id, gt))

        for feat in sample_precomputed:
            all_precomputed_info.append({
                "video_grid_thw": feat["video_grid_thw"].squeeze(0),  # [3]
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

    # Build text inputs with precomputed grid info (no video decoding)
    proc_out = processor.call_with_precomputed(
        texts,
        precomputed_info=all_precomputed_info,
        padding=True,
        return_tensors="pt",
    )

    feature_inputs = torch.cat(all_visual_tokens, dim=0)
    video_grid_thw = torch.cat(all_grid_thw, dim=0)

    if all_deepstack is not None:
        deepstack_feature_inputs = [torch.cat(layer, dim=0) for layer in all_deepstack]
    else:
        deepstack_feature_inputs = None

    print(f"[DEBUG assemble] videos={len(all_visual_tokens)}  "
          f"per-video tokens={[t.shape[0] for t in all_visual_tokens]}  "
          f"feature_inputs={feature_inputs.shape}  "
          f"video_grid_thw={video_grid_thw.tolist()}  "
          f"input_ids={proc_out['input_ids'].shape}")

    inputs = {
        **proc_out,
        "video_grid_thw": video_grid_thw,
        "feature_inputs": feature_inputs,
        "deepstack_feature_inputs": deepstack_feature_inputs,
    }
    return inputs, batch_meta


def assemble_online_batch(batch, processor):
    """
    Build model inputs by decoding video frames on-the-fly (fallback path).
    """
    texts = []
    all_images = []
    all_videos = []
    all_video_metadatas = []
    batch_meta = []

    for item in batch:
        sample_id = item["id"]
        video_paths = item["video"]
        prompt = item["conversations"][0]["value"]
        gt = item["conversations"][1]["value"].strip().upper()

        messages = build_messages(video_paths, prompt)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if video_inputs is not None:
            video_tensors, video_metadatas = zip(*video_inputs)
            all_videos.extend(video_tensors)
            all_video_metadatas.extend(video_metadatas)
        if image_inputs:
            all_images.extend(image_inputs)

        texts.append(text)
        batch_meta.append((sample_id, gt))

    vkw = dict(video_kwargs or {})
    vkw.pop("fps", None)

    proc_out = processor(
        text=texts,
        images=all_images or None,
        videos=all_videos or None,
        video_metadata=all_video_metadatas or None,
        padding=True,
        return_tensors="pt",
        do_resize=False,
        return_mm_token_type_ids=True,
        **vkw,
    )
    return proc_out, batch_meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("results", exist_ok=True)

    print("Loading model...")
    model = Qwen3VLWithOfflineFeatures.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()

    processor = Qwen3VLProcessorWithPrecomputed.from_pretrained(MODEL_NAME)
    processor.tokenizer.padding_side = "left"

    with open(DATA_PATH) as f:
        data = json.load(f)

    all_data = data[:LIMIT] if LIMIT > 0 else data
    subset = [
        item for item in all_data
        if item["video"] and all(
            load_precomputed_for_video(vp) is not None for vp in item["video"]
        )
    ]
    print(f"Samples with precomputed features: {len(subset)} / {len(all_data)}")
    total = correct = 0

    fsa_results = []

    with open(OUT_PATH, "w") as fout:
        for batch_start in tqdm(range(0, len(subset), BATCH_SIZE)):
            batch = subset[batch_start: batch_start + BATCH_SIZE]

            # -------------------------------------------------------
            # Try precomputed path first; fall back to online decoding
            # -------------------------------------------------------
            try:
                inputs, batch_meta = assemble_offline_batch(batch, processor)
                use_offline = True
            except RuntimeError as missing_err:
                print(f"\n[INFO] Falling back to online decoding: {missing_err}")
                try:
                    inputs, batch_meta = assemble_online_batch(batch, processor)
                    use_offline = False
                except Exception as e:
                    for item in batch:
                        fout.write(json.dumps({"id": item["id"], "error": str(e)}) + "\n")
                    print(f"[ERROR] batch {batch_start}: {e}")
                    continue

            try:
                inputs = {
                    k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }
                if isinstance(inputs.get("deepstack_feature_inputs"), list):
                    inputs["deepstack_feature_inputs"] = [
                        t.to(model.device) for t in inputs["deepstack_feature_inputs"]
                    ]

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                    )

                generated_ids_trimmed = [
                    out[len(inp):]
                    for inp, out in zip(inputs["input_ids"], generated_ids)
                ]
                responses = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                for (sample_id, gt), response in zip(batch_meta, responses):
                    task = get_task(sample_id)
                    pred = extract_pred(response, task)
                    record = {
                        "id": sample_id,
                        "pred_raw": response,
                        "gt": gt,
                        "pred": pred,
                        "offline": use_offline,
                    }

                    if task == "FSA":
                        iou = temporal_iou(pred, gt) if pred else 0.0
                        fsa_results.append({"id": sample_id, "iou": round(iou, 3)})
                        record["iou"] = round(iou, 3)
                    elif task == "PSS":
                        is_correct = pred == gt.strip()
                        record["correct"] = is_correct
                        correct += int(is_correct)
                        total += 1
                    else:
                        is_correct = pred == normalize_gt(gt, task)
                        record["correct"] = is_correct
                        correct += int(is_correct)
                        total += 1
                    fout.write(json.dumps(record) + "\n")

                acc = correct / total if total > 0 else 0.0
                last_id, last_gt = batch_meta[-1]
                print(
                    f"[{total}] id={last_id} gt={last_gt} "
                    f"acc={acc:.3f} {'(offline)' if use_offline else '(online)'}"
                )

            except Exception as e:
                for sample_id, _ in batch_meta:
                    fout.write(json.dumps({"id": sample_id, "error": str(e)}) + "\n")
                print(f"[ERROR] batch {batch_start}: {e}")

    if total > 0:
        print(f"\nMC/PSS accuracy: {correct}/{total} = {correct / total:.3f}")
    if fsa_results:
        mean_iou = sum(p["iou"] for p in fsa_results) / len(fsa_results)
        print(f"FSA mean IoU: {mean_iou:.3f}  ({len(fsa_results)} samples)")
    if not total and not fsa_results:
        print("\nNo samples evaluated.")
    print(f"Results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
