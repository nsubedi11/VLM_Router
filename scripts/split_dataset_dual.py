

import json
import random
import os

# setting
input_path = "data/processed/all_6tasks.json"
output_base = "splits"
random_seed = 42

# load
with open(input_path, "r") as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# shuffle
random.seed(random_seed)
random.shuffle(data)


def split_data(data, train_ratio, val_ratio, test_ratio, name):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # making directory
    out_dir = os.path.join(output_base, name)
    os.makedirs(out_dir, exist_ok=True)

    # save
    with open(f"{out_dir}/train.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open(f"{out_dir}/val.json", "w") as f:
        json.dump(val_data, f, indent=2)

    with open(f"{out_dir}/test.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"\n[{name}]")
    print(f"Train: {len(train_data)}")
    print(f"Val: {len(val_data)}")
    print(f"Test: {len(test_data)}")

# execute
split_data(data, 0.8, 0.1, 0.1, "split_80_10_10")
split_data(data, 0.7, 0.15, 0.15, "split_70_15_15")