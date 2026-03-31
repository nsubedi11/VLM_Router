# scripts/video_stats.py
#
# Collect duration and spatial dimensions for every unique video across all
# splits in splits/split_80_10_10.
#
# Outputs:
#   results/video_stats.jsonl  — one record per video
#   results/video_stats_summary.txt — aggregate statistics
#
# Run from the project root:
#   python scripts/video_stats.py

import os
import sys
import json
import subprocess
import numpy as np
from fractions import Fraction
from tqdm import tqdm

SPLITS_DIR = "splits/split_80_10_10"
OUT_JSONL = "results/video_stats.jsonl"
OUT_SUMMARY = "results/video_stats_summary.txt"


def collect_videos() -> list[str]:
    seen = set()
    paths = []
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


def get_video_stats(video_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        video_path,
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    info = json.loads(out)

    video_stream = next(
        s for s in info["streams"] if s["codec_type"] == "video"
    )

    width  = int(video_stream["width"])
    height = int(video_stream["height"])
    fps    = float(Fraction(video_stream["r_frame_rate"]))

    # duration: prefer stream-level, fall back to format-level
    duration = float(
        video_stream.get("duration")
        or info.get("format", {}).get("duration", 0)
    )

    nb_frames = video_stream.get("nb_frames")
    frames = int(nb_frames) if nb_frames else round(duration * fps)

    return {
        "path": video_path,
        "duration_s": round(duration, 3),
        "fps": round(fps, 3),
        "frames": frames,
        "height": height,
        "width": width,
    }


def summarize(records: list[dict]) -> str:
    durations = np.array([r["duration_s"] for r in records])
    heights   = np.array([r["height"]     for r in records])
    widths    = np.array([r["width"]      for r in records])
    fpss      = np.array([r["fps"]        for r in records])

    def stats(arr, label, unit=""):
        return (
            f"{label}:\n"
            f"  min={arr.min():.2f}{unit}  max={arr.max():.2f}{unit}  "
            f"mean={arr.mean():.2f}{unit}  median={np.median(arr):.2f}{unit}  "
            f"p25={np.percentile(arr,25):.2f}{unit}  p75={np.percentile(arr,75):.2f}{unit}\n"
        )

    lines = [
        f"Total videos: {len(records)}\n",
        stats(durations, "Duration",  unit="s"),
        stats(heights,   "Height",    unit="px"),
        stats(widths,    "Width",     unit="px"),
        stats(fpss,      "FPS"),
    ]

    # resolution breakdown
    from collections import Counter
    res_counts = Counter(f"{r['height']}x{r['width']}" for r in records)
    lines.append("Top resolutions:\n")
    for res, cnt in res_counts.most_common(10):
        lines.append(f"  {res}: {cnt}\n")

    return "".join(lines)


def main():
    os.makedirs("results", exist_ok=True)

    video_paths = collect_videos()
    print(f"Found {len(video_paths)} unique videos in {SPLITS_DIR}")

    records = []
    errors = []

    with open(OUT_JSONL, "w") as fout:
        for vp in tqdm(video_paths, desc="Reading video metadata"):
            try:
                stats = get_video_stats(vp)
                records.append(stats)
                fout.write(json.dumps(stats) + "\n")
            except Exception as e:
                err = {"path": vp, "error": str(e)}
                errors.append(err)
                fout.write(json.dumps(err) + "\n")
                print(f"\n[ERROR] {vp}: {e}")

    summary = summarize(records)
    print("\n" + summary)

    with open(OUT_SUMMARY, "w") as f:
        f.write(summary)
        if errors:
            f.write(f"\nFailed ({len(errors)}):\n")
            for e in errors:
                f.write(f"  {e['path']}: {e['error']}\n")

    print(f"Saved stats  → {OUT_JSONL}")
    print(f"Saved summary → {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
