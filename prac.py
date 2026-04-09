import torch


dir = "/scratch/rai/vast1/alhalah/users/nikesh/qwen3vl_proj/features/qwen_video/2607fdf555a0968ca015febedabbd797.pt"

feat = torch.load(dir)
print(feat.keys())
print(feat["visual_tokens"].shape)
print(feat["deepstack_features"][0].shape)
print(feat["video_grid_thw"])

