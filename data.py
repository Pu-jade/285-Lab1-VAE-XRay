import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def discover_images(folder):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")
    images = []
    for path in folder.rglob("*"):
        parts = path.relative_to(folder).parts
        hidden = any(part.startswith(".") or part == "__MACOSX" for part in parts)
        if hidden or not path.is_file():
            continue
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            images.append(str(path))
    return sorted(images)


def patient_key(path):
    name = Path(path).stem.lower()
    match = re.search(r"person(\d+)_(bacteria|virus)", name)
    if match:
        return "person:" + match.group(1) + ":" + match.group(2)
    match = re.match(r"(normal2-im|im)-(\d+)", name)
    if match:
        return match.group(1) + ":" + match.group(2)
    return "file:" + str(Path(path).resolve())


def prepare_splits(data_dir, seed=42):
    root = Path(data_dir).expanduser().resolve()
    train_files = discover_images(root / "train")
    test_files = discover_images(root / "test")
    if not train_files or not test_files:
        raise ValueError("train/ and test/ must contain images.")
    test_patients = {patient_key(path) for path in test_files}
    groups = {}
    for path in train_files:
        patient = patient_key(path)
        if patient not in test_patients:
            groups.setdefault(patient, []).append(path)
    patients = sorted(groups)
    if len(patients) < 2:
        raise ValueError("Need at least two training patient groups.")
    random.Random(seed).shuffle(patients)
    val_count = max(1, round(0.1 * len(patients)))
    val_patients = set(patients[:val_count])
    splits = {"train": [], "val": [], "test": test_files}
    for patient, paths in groups.items():
        split = "val" if patient in val_patients else "train"
        splits[split].extend(paths)
    return splits


class ChestXrayDataset(Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
            pixels = np.array(image, dtype=np.float32) / 255.0
        return torch.from_numpy(pixels).unsqueeze(0)
