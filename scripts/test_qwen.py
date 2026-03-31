import torch
import json
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

model_name = "Qwen/Qwen3-VL-2B-Instruct"

print("CUDA available:", torch.cuda.is_available())

print("Loading model...")
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

print("Model device:", model.device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("Loading sample...")
with open("data/processed/sample_10.json", "r") as f:
    data = json.load(f)

item = data[0]
prompt = item["conversations"][0]["value"]

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt}
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(
    text=[text],
    return_tensors="pt",
    padding=True,
)

inputs = {k: v.to(model.device) for k, v in inputs.items()}

print("Generating...")
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
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