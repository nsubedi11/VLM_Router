import json
import os
import random

QA_DIR = "data/CrossVid_hf/QA"
VIDEO_ROOT = "/scratch/general/vast/u1209255/CrossVid_Dataset/videos"

SELECTED_FILES = ["NC", "CC", "PEA", "PI", "FSA", "PSS"]

OUTPUT_ALL = "data/processed/all_6tasks.json"
OUTPUT_SAMPLE = "data/processed/sample_10.json"

SEED = 42


def video_tags(num_videos):
    return "\n".join(f"Video {i+1}: <video>" for i in range(num_videos))


def make_item(task_name, item_id, abs_videos, prompt, answer):
    return {
        "id": f"{task_name}_{item_id}",
        "video": abs_videos,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": str(answer).strip()},
        ],
    }


def convert_cc(task_name, item):
    videos = [os.path.join(VIDEO_ROOT, v) for v in item["videos"]]
    opts = "\n".join(o.strip() for o in item["options"])
    prompt = (
        f"{video_tags(len(videos))}\n"
        "Provide you four videos and a single-choice question with only one correct option.\n"
        "Watch the videos carefully, and think about the question based on the information from these videos.\n"
        "Select one answer choice, and only output the capital letter of your choice.\n\n"
        f"Question:\n{item['question'].strip()}\n\n"
        f"Options:\n{opts}\n\n"
        "Your answer:"
    )
    return make_item(task_name, item["id"], videos, prompt, item["answer"])


def convert_nc(task_name, item):
    videos = [os.path.join(VIDEO_ROOT, v) for v in item["videos"]]
    opts = "\n".join(o.strip() for o in item["options"])
    prompt = (
        f"{video_tags(len(videos))}\n"
        "Provide you four videos and a single-choice question with only one correct option.\n"
        "Watch the videos carefully, and think about the question based on the information from the four videos.\n"
        "Select one answer choice, and only output the capital letter of your choice.\n\n"
        f"Question:\n{item['question'].strip()}\n\n"
        f"Options:\n{opts}\n\n"
        "Your answer:"
    )
    return make_item(task_name, item["id"], videos, prompt, item["answer"])


def convert_pea(task_name, item):
    videos = [os.path.join(VIDEO_ROOT, v) for v in item["videos"]]
    opts = "\n".join(o.strip() for o in item["options"])
    prompt = (
        f"{video_tags(len(videos))}\n"
        "Provide you three videos assembling the same toy car and a single-choice question.\n"
        "In addition, provide you four predefined error types that may assist you answer.\n"
        "- wrong order: this action is an ordering mistake.\n"
        "- previous one is mistake: this action is also an ordering mistake but is caused by the preceding ordering mistakes in the context.\n"
        "- shouldn't have happened: this action is unnecessary in the assembly.\n"
        "- wrong position: the two parts are not attached at their correct position.\n"
        "Watch the videos carefully, and think about the question based on the information from these videos.\n"
        "Select one answer choice, and only output the capital letter of your choice.\n\n"
        f"Question:\n{item['question'].strip()}\n\n"
        f"Options:\n{opts}\n\n"
        "Your answer:"
    )
    return make_item(task_name, item["id"], videos, prompt, item["answer"])


def convert_pi(task_name, item):
    video = os.path.join(VIDEO_ROOT, item["video"])
    opts = "\n".join(o.strip() for o in item["options"])
    prompt = (
        f"Video 1: <video>\n"
        "Provide you the beginning and the ending of a movie clip, what is most likely to happen in the middle?\n"
        "Watch the video segments carefully, and think about the question based on the context information.\n"
        "Select one answer choice, and only output the capital letter of your choice.\n\n"
        f"Options:\n{opts}\n\n"
        "Your answer:"
    )
    return make_item(task_name, item["id"], [video], prompt, item["answer"])


def convert_fsa(task_name, item):
    video_a = os.path.join(VIDEO_ROOT, item["video A"])
    video_b = os.path.join(VIDEO_ROOT, item["video B"])
    ref = item["ref_segment"]
    prompt = (
        f"{video_tags(2)}\n"
        f"Provide you two cooking videos, which step in Video 2 is functionally equivalent to the step shown between {ref[0]}s and {ref[1]}s in Video 1?\n"
        "Watch the two videos carefully, and think about the question based on the information of the two videos.\n"
        'Only output a time interval in seconds and separate the beginning and ending time with a comma, e.g., "15,23".\n\n'
        "Your answer:"
    )
    ans = item["answer"]
    return make_item(task_name, item["id"], [video_a, video_b], prompt, f"{ans[0]},{ans[1]}")


def convert_pss(task_name, item):
    video = os.path.join(VIDEO_ROOT, item["video"])
    segs = item["segments"]
    n = len(segs)
    prompt = (
        f"Video 1: <video>\n"
        f"Provide you {n} shuffled segments of a cooking video, what's the correct order of these segments?\n"
        "Watch the segments carefully, and think about the question based on the relationship between these segments.\n"
        'Only output the correct segment number sequence separated by "->", e.g., "2->3->1->4".\n\n'
        "Your answer:"
    )
    return make_item(task_name, item["id"], [video], prompt, item["answer"])


CONVERTERS = {
    "CC": convert_cc,
    "NC": convert_nc,
    "PEA": convert_pea,
    "PI": convert_pi,
    "FSA": convert_fsa,
    "PSS": convert_pss,
}


def load_and_convert():
    all_data = []

    for task in SELECTED_FILES:
        json_path = os.path.join(QA_DIR, f"{task}.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        converter = CONVERTERS[task]
        converted = [converter(task, item) for item in data]

        # Deduplicate by (video, prompt, answer) — drop exact content duplicates
        seen = set()
        deduped = []
        for item in converted:
            key = (tuple(item["video"]), item["conversations"][0]["value"], item["conversations"][1]["value"])
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        if len(deduped) < len(converted):
            print(f"[INFO] {task}: {len(converted)} examples ({len(converted) - len(deduped)} duplicates removed)")
        else:
            print(f"[INFO] {task}: {len(converted)} examples")
        all_data.extend(deduped)

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
    os.makedirs("data/processed", exist_ok=True)

    all_data = load_and_convert()
    print(f"[INFO] Total: {len(all_data)} examples")

    missing = validate_video_paths(all_data)
    print(f"[INFO] Missing video files: {len(missing)}")
    if missing:
        print("[INFO] First 20 missing paths:")
        for m in missing[:20]:
            print(m)

    save_json(OUTPUT_ALL, all_data)
    save_json(OUTPUT_SAMPLE, all_data[:10])
    print("[DONE]")


if __name__ == "__main__":
    main()