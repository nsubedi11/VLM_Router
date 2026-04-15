# scripts/dataset.py
#
# VideoQADataset: loads precomputed visual features + tokenises each
# conversation (question + answer) into model inputs.

import os
import sys
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from vision_cache import video_feat_path, load_precomputed_video
from models.qwen3_vl import Qwen3VLProcessorWithPrecomputed


RESIZED_HEIGHT = 480
FPS            = 0.5


class VideoQADataset(Dataset):
    """
    Each sample contains everything the model needs for one QA example:

        input_ids       [1, L]   tokenised user turn + assistant answer
        attention_mask  [1, L]
        answer_tokens   [n]      answer token IDs (used to build training labels)
        video_grid_thw          list of [1, 3] tensors — one per video
        feature_inputs          list of visual token tensors — one per video
        deepstack_inputs        list[list[Tensor]] (layer × video) or None

    The dataset caches processed samples to disk so the first run (which
    calls the processor for every sample) only happens once.
    """

    def __init__(
        self,
        data_path:    str,
        feat_dir:     str,
        processor:    Qwen3VLProcessorWithPrecomputed,
        pad_token_id: int,
        cache_path:   str | None = None,
    ):
        if cache_path and os.path.exists(cache_path):
            print(f"Loading dataset cache: {cache_path}")
            self.samples = torch.load(cache_path, weights_only=False)
            print(f"  {len(self.samples)} samples ready")
            return

        with open(data_path) as f:
            data = json.load(f)

        self.samples: list[dict] = []
        skipped = 0

        for item in tqdm(data, desc=f"Building {os.path.basename(data_path)}"):
            video_paths = item.get("video", [])
            if not video_paths:
                skipped += 1
                continue

            sample = self._process_item(item, video_paths, feat_dir, processor)
            if sample is None:
                skipped += 1
                continue
            self.samples.append(sample)

        print(f"  {len(self.samples)} samples built  ({skipped} skipped — missing features or seq_len > 30k)")
        print(f"  NOTE: visual features are loaded from disk per step (not cached in RAM)")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.samples, cache_path)
            print(f"  Saved to cache: {cache_path}")

    # ------------------------------------------------------------------

    def _process_item(self, item, video_paths, feat_dir, processor):
        """
        Build one dataset sample from a raw JSON item.
        Returns None if any video's feature cache file is missing.

        Visual feature tensors are NOT stored here — only the cache file paths
        are saved, so the dataset stays small in memory.  Tensors are loaded
        per step inside __getitem__.
        """
        precomputed_info = []
        feat_paths       = []   # paths to .pt cache files — loaded on demand

        for vp in video_paths:
            fpath = video_feat_path(vp, feat_dir=feat_dir)
            feat  = load_precomputed_video(fpath)
            if feat is None:
                return None

            precomputed_info.append({
                "video_grid_thw": feat["video_grid_thw"].squeeze(0),
                "frames_indices": feat["frames_indices"],
                "fps":            feat["fps"],
            })
            feat_paths.append(fpath)

        # Build the tokenised user turn
        prompt   = item["conversations"][0]["value"]
        messages = self._build_messages(video_paths, prompt)
        text     = processor.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        proc_out = processor.call_with_precomputed(
            text, precomputed_info=precomputed_info,
            padding=False, return_tensors="pt",
        )

        # Tokenise the answer separately to know its exact token IDs
        answer        = item["conversations"][1]["value"].strip()
        answer_tokens = processor.tokenizer(
            answer + processor.tokenizer.eos_token,
            add_special_tokens=False, return_tensors="pt",
        )["input_ids"][0]   # [n]

        # Full sequence = user turn + answer
        full_ids  = torch.cat([proc_out["input_ids"][0], answer_tokens]).unsqueeze(0)
        full_mask = torch.ones(1, full_ids.shape[1], dtype=torch.long)

        if full_ids.shape[1] > 30_000:
            return None

        return {
            "input_ids":      full_ids,
            "attention_mask": full_mask,
            "answer_tokens":  answer_tokens,
            "feat_paths":     feat_paths,   # list of strings — no large tensors
        }

    @staticmethod
    def _build_messages(video_paths: list[str], prompt: str) -> list:
        content = []
        for i, vp in enumerate(video_paths):
            content.append({"type": "text",  "text": f"This is Video {i + 1}."})
            content.append({
                "type":           "video",
                "video":          vp,
                "resized_height": RESIZED_HEIGHT,
                "fps":            FPS,
                "min_pixels":     16 * 28 * 28,
                "max_pixels":     128 * 28 * 28,
            })
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load visual feature tensors from disk on demand (keeps RAM usage low)
        feature_inputs = []
        grid_thw       = []
        deepstack      = None

        for fpath in sample["feat_paths"]:
            feat = load_precomputed_video(fpath)

            feature_inputs.append(feat["visual_tokens"])
            grid_thw.append(feat["video_grid_thw"])

            ds = feat.get("deepstack_features") or []
            if ds:
                if deepstack is None:
                    deepstack = [[] for _ in range(len(ds))]
                for layer_idx, layer_feat in enumerate(ds):
                    deepstack[layer_idx].append(layer_feat)

        return {
            "input_ids":       sample["input_ids"],
            "attention_mask":  sample["attention_mask"],
            "answer_tokens":   sample["answer_tokens"],
            "video_grid_thw":  grid_thw,
            "feature_inputs":  feature_inputs,
            "deepstack_inputs": deepstack,
        }


def collate_fn(batch: list) -> dict:
    """
    DataLoader collate function.
    BATCH_SIZE must be 1 because each sample has a different number of
    video tokens — padding them together is non-trivial and unnecessary.
    """
    assert len(batch) == 1, "Only BATCH_SIZE=1 is supported"
    return batch[0]
