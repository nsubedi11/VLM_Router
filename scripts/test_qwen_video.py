import torch
import json
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

model_name = "Qwen/Qwen3-VL-2B-Instruct"

print("CUDA available:", torch.cuda.is_available())

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_name)

print("Model device:", model.device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Loading sample...")
with open("data/processed/sample_10.json", "r") as f:
    data = json.load(f)

item = data[0]
video_paths = item["video"]
prompt = item["conversations"][0]["value"]
gt_answer = item["conversations"][1]["value"]

print("Ground truth:", gt_answer)
print("Video paths:")
for i, vp in enumerate(video_paths, 1):
    print(f"  Video {i}: {vp}")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": f"file://{video_paths[0]}", "min_pixels": 96 * 96, "max_pixels": 160 * 160, "fps": 0.1},
            {"type": "text", "text": "This is Video 1."},

            {"type": "video", "video": f"file://{video_paths[1]}", "min_pixels": 96 * 96, "max_pixels": 160 * 160, "fps": 0.1},
            {"type": "text", "text": "This is Video 2."},

            {"type": "video", "video": f"file://{video_paths[2]}", "min_pixels": 96 * 96, "max_pixels": 160 * 160, "fps": 0.1},
            {"type": "text", "text": "This is Video 3."},

            {"type": "video", "video": f"file://{video_paths[3]}", "min_pixels": 96 * 96, "max_pixels": 160 * 160, "fps": 0.1},
            {"type": "text", "text": "This is Video 4."},

            {"type": "text", "text": prompt + "\n\nAnswer with only one letter: A, B, C, or D."},
        ],
    }
]

print("Preparing inputs...")
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs, video_kwargs = process_vision_info(
    messages,
    image_patch_size=16,
    return_video_kwargs=True,
    return_video_metadata=True,
)

if video_inputs is not None:
    video_inputs, video_metadatas = zip(*video_inputs)
    video_inputs = list(video_inputs)
    video_metadatas = list(video_metadatas)
else:
    video_metadatas = None

video_kwargs = dict(video_kwargs or {})
print("Raw video_kwargs:", video_kwargs)

# preventing fps type conflict during deal with multi-video
video_kwargs.pop("fps", None)

inputs = processor(
    text=text,
    images=image_inputs,
    videos=video_inputs,
    video_metadata=video_metadatas,
    return_tensors="pt",
    do_resize=False,
    **video_kwargs,
)

inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
inputs.pop("token_type_ids", None)

print("Generating...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
    )

generated_ids = [
    out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)
]

response = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True,
)

print("\nPrompt:")
print(prompt)
print("\nResponse:")
print(response[0])
print("\nGround truth:")
print(gt_answer)