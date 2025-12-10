#!/usr/bin/env python3
import shutil
from pathlib import Path
import random

random.seed(42)

dataset_path = Path("datasets/game_state_1")
train_split = 0.8

for class_name in ["play", "no-play"]:
    class_dir = dataset_path / class_name
    videos = sorted(list(class_dir.glob("*.mp4")))

    random.shuffle(videos)

    split_idx = int(len(videos) * train_split)
    train_videos = videos[:split_idx]
    test_videos = videos[split_idx:]

    (dataset_path / "train" / class_name).mkdir(parents=True, exist_ok=True)
    (dataset_path / "test" / class_name).mkdir(parents=True, exist_ok=True)

    for video in train_videos:
        shutil.copy2(video, dataset_path / "train" / class_name / video.name)

    for video in test_videos:
        shutil.copy2(video, dataset_path / "test" / class_name / video.name)

    print(f"{class_name}: {len(train_videos)} train, {len(test_videos)} test")

print("Dataset split complete!")
