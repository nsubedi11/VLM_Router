# scripts/test_qwen_video.py
#
# Evaluate Qwen3-VL on the full video QA test split.
# Reuses helpers from eval_qwen_video.py.

import os
import sys
import json
import argparse
import torch
from tqdm import tqdm

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from eval_qwen_video import (
    _to_device,
    get_task,
    extract_pred,
    normalize_gt,
    temporal_iou,
    build_messages,
    load_precomputed_for_video,
    assemble_offline_batch,
    assemble_online_batch,
)
from vision_cache import video_feat_path, FEAT_DIR
from models.qwen3_vl import Qwen3VLWithOfflineFeatures, Qwen3VLProcessorWithPrecomputed
from models.resolution_router import ResolutionRouter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
DATA_PATH  = "splits/split_70_15_15/test.json"
OUT_PATH   = "results/qwen_video_test.jsonl"
MAX_NEW_TOKENS = 96
BATCH_SIZE = 1


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL on the full test split.")
    parser.add_argument("--use_fixed_pooling", type=int, default=-1,
                        help="Fix pool_level for all samples (-1 = use router, 0/1/2 = fixed level).")
    parser.add_argument("--out_path", type=str, default=OUT_PATH,
                        help="Path for the output JSONL file.")
    parser.add_argument('-c',"--ckpt_path", type=str, default=None,
                        help="Path to checkpoint dir containing lora/ and router.pt.")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    print("Loading model...")
    model = Qwen3VLWithOfflineFeatures.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    if args.ckpt_path is not None:
        lora_dir = os.path.join(args.ckpt_path, "lora")
        if HAS_PEFT and os.path.isdir(lora_dir):
            print(f"Loading LoRA weights from {lora_dir}")
            model = PeftModel.from_pretrained(model, lora_dir)
            model = model.merge_and_unload()
        elif not HAS_PEFT:
            print("[WARNING] peft not installed — skipping LoRA loading")

    model.eval()

    _embed_device = next(model.parameters()).device
    router = ResolutionRouter(input_dim=2048).to(_embed_device)

    if args.ckpt_path is not None:
        router_pt = os.path.join(args.ckpt_path, "router.pt")
        if os.path.isfile(router_pt):
            print(f"Loading router weights from {router_pt}")
            router.load_state_dict(torch.load(router_pt, map_location=_embed_device))

    router.eval()
    model.resolution_router = router

    processor = Qwen3VLProcessorWithPrecomputed.from_pretrained(MODEL_NAME)
    processor.tokenizer.padding_side = "left"
    model._debug_tokenizer = processor.tokenizer

    with open(DATA_PATH) as f:
        data = json.load(f)
    
    print(f"checking precomputed features for {len(data)} samples...")

    subset = [
        item for item in data
        if item["video"] and all(
            os.path.exists(video_feat_path(vp, feat_dir=FEAT_DIR)) for vp in item["video"]
        )
    ]
    print(f"Samples with precomputed features: {len(subset)} / {len(data)}")
    total = correct = 0
    fsa_results = []
    pool_level_counts = {}

    with open(args.out_path, "w") as fout:
        for batch_start in tqdm(range(0, len(subset), BATCH_SIZE)):
            batch = subset[batch_start: batch_start + BATCH_SIZE]

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
                inputs = {k: _to_device(v, model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        use_fixed_pooling=args.use_fixed_pooling,
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

                pool_level = getattr(model, "_last_pool_level", None)
                if pool_level is not None:
                    pool_level_counts[pool_level] = pool_level_counts.get(pool_level, 0) + 1
                for (sample_id, gt), response in zip(batch_meta, responses):
                    task = get_task(sample_id)
                    pred = extract_pred(response, task)
                    record = {
                        "id": sample_id,
                        "pred_raw": response,
                        "gt": gt,
                        "pred": pred,
                        "offline": use_offline,
                        "pool_level": pool_level,
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

        # Write summary to results file
        summary = {"type": "summary"}
        if total > 0:
            summary["mc_pss_accuracy"] = {
                "correct": correct,
                "total": total,
                "accuracy": round(correct / total, 3)
            }
        if fsa_results:
            mean_iou = sum(p["iou"] for p in fsa_results) / len(fsa_results)
            summary["fsa_mean_iou"] = {
                "mean_iou": round(mean_iou, 3),
                "num_samples": len(fsa_results)
            }
        if pool_level_counts:
            summary["pool_level_counts"] = {str(k): v for k, v in sorted(pool_level_counts.items())}
        fout.write(json.dumps(summary) + "\n")

    if total > 0:
        print(f"\nMC/PSS accuracy: {correct}/{total} = {correct / total:.3f}")
    if fsa_results:
        mean_iou = sum(p["iou"] for p in fsa_results) / len(fsa_results)
        print(f"FSA mean IoU: {mean_iou:.3f}  ({len(fsa_results)} samples)")
    if not total and not fsa_results:
        print("\nNo samples evaluated.")
    print(f"Results saved to {args.out_path}")


if __name__ == "__main__":
    main()
