import json
import os
import random

QA_DIR = "/scratch/general/vast/u1209255/CrossVid_Dataset/QA"
VIDEO_ROOT = "/scratch/general/vast/u1209255/CrossVid_Dataset/videos"

SELECTED_FILES = ["NC", "CC", "PEA", "PI", "FSA", "PSS"]

OUTPUT_ALL = "/scratch/general/vast/u1209255/qwen3vl_project/data/processed/all_6tasks.json"
OUTPUT_TRAIN = "/scratch/general/vast/u1209255/qwen3vl_project/data/processed/train.json"
OUTPUT_VAL = "/scratch/general/vast/u1209255/qwen3vl_project/data/processed/val.json"
OUTPUT_TEST = "/scratch/general/vast/u1209255/qwen3vl_project/data/processed/test.json"
OUTPUT_SAMPLE = "/scratch/general/vast/u1209255/qwen3vl_project/data/processed/sample_10.json"

SEED = 42


def build_prompt(question, options, num_videos):
    lines = []
    for i in range(num_videos):
        lines.append(f"Video {i+1}: <video>")

    lines.append(question.strip())

    if options:
        lines.append("Choices:")
        for opt in options:
            lines.append(opt.strip())

    return "\n".join(lines)


def convert_item(task_name, item):
    raw_videos = item.get("videos", [])
    abs_videos = [os.path.join(VIDEO_ROOT, v) for v in raw_videos]

    prompt = build_prompt(
        question=item.get("question", ""),
        options=item.get("options", []),
        num_videos=len(abs_videos),
    )

    new_item = {
        "id": f"{task_name}_{item['id']}",
        "video": abs_videos,
        "conversations": [
            {
                "from": "human",
                "value": prompt
            },
            {
                "from": "gpt",
                "value": str(item.get("answer", "")).strip()
            }
        ]
    }
    return new_item


def load_and_convert():
    all_data = []

    for task in SELECTED_FILES:
        json_path = os.path.join(QA_DIR, f"{task}.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        print(f"[INFO] Loaded {task}: {len(data)} examples")

        for item in data:
            all_data.append(convert_item(task, item))

    return all_data


def validate_video_paths(data):
    missing = []
    for item in data:
        for vp in item["video"]:
            if not os.path.exists(vp):
                missing.append(vp)

    return missing


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[SAVED] {path} ({len(obj) if isinstance(obj, list) else 'object'})")


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    random.shuffle(data)
    n = len(data)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def main():
    random.seed(SEED)

    os.makedirs(os.path.dirname(OUTPUT_ALL), exist_ok=True)

    all_data = load_and_convert()

    print(f"[INFO] Total converted examples: {len(all_data)}")

    missing = validate_video_paths(all_data)
    print(f"[INFO] Missing video files: {len(missing)}")
    if missing:
        print("[INFO] First 20 missing paths:")
        for m in missing[:20]:
            print(m)

    save_json(OUTPUT_ALL, all_data)
    save_json(OUTPUT_SAMPLE, all_data[:10])

    train_data, val_data, test_data = split_data(all_data)
    save_json(OUTPUT_TRAIN, train_data)
    save_json(OUTPUT_VAL, val_data)
    save_json(OUTPUT_TEST, test_data)

    print("[DONE] Conversion finished.")


if __name__ == "__main__":
    main()