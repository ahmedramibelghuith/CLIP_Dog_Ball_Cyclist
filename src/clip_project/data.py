"""Validation et partition commune, sans dépendance TensorFlow."""
import csv
import random
from collections import defaultdict
from pathlib import Path


def read_rows(csv_path, image_root):
    with Path(csv_path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"image_path", "caption", "label"} <= set(reader.fieldnames or []):
            raise ValueError("Le CSV doit contenir image_path, caption et label.")
        rows = list(reader)
    if not rows:
        raise ValueError("Le CSV est vide.")
    root = Path(image_root).resolve()
    seen = set()
    for row in rows:
        path = (root / row["image_path"]).resolve()
        if path in seen:
            raise ValueError(f"Image répétée : {path}. Une paire par image est requise.")
        if not path.is_file():
            raise FileNotFoundError(f"Image absente : {path}")
        if not row["caption"].strip() or not row["label"].strip():
            raise ValueError("Description et classe doivent être renseignées.")
        seen.add(path)
        row["image_path"] = str(path)
    return rows


def split_rows(rows, seed=42):
    """Partition stratifiée 80/10/10, réutilisée pour chaque modèle."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["label"]].append(row)
    result = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for label in sorted(groups):
        group = groups[label][:]
        if len(group) < 10:
            raise ValueError(f"Au moins 10 images sont nécessaires pour la classe {label}.")
        rng.shuffle(group)
        n_val = n_test = max(1, len(group) // 10)
        result["val"].extend(group[:n_val])
        result["test"].extend(group[n_val:n_val + n_test])
        result["train"].extend(group[n_val + n_test:])
    for part in result.values():
        rng.shuffle(part)
    return result


def write_rows(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_path", "caption", "label"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
