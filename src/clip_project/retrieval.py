"""Index associé au modèle, recherche et Recall@K sur une galerie tenue à l'écart."""
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from .data import read_rows
from .pipeline import load_bundle, load_image


def fingerprint(run):
    digest = hashlib.sha256()
    for name in ("clip.keras", "config.json"):
        with (Path(run) / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def embed(model, tokenizer, rows, batch_size=32):
    paths = [r["image_path"] for r in rows]
    captions = [r["caption"] for r in rows]
    images = tf.data.Dataset.from_tensor_slices(paths).map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    image_vectors = model.get_layer("image_encoder").predict(images.batch(batch_size), verbose=0)
    texts = tf.data.Dataset.from_tensor_slices(captions).batch(batch_size)
    text_encoder = model.get_layer("text_encoder")
    text_vectors = np.concatenate([text_encoder(tokenizer(batch), training=False).numpy() for batch in texts])
    return image_vectors, text_vectors


def build_index(args):
    model, tokenizer = load_bundle(args.run)
    rows = read_rows(args.csv, args.image_root)
    images, texts = embed(model, tokenizer, rows, args.batch_size)
    path = Path(args.run) / "index.npz"
    np.savez_compressed(path, images=images, texts=texts,
                        paths=np.array([r["image_path"] for r in rows]),
                        captions=np.array([r["caption"] for r in rows]),
                        fingerprint=np.array(fingerprint(args.run)))
    print(f"Index sauvegardé : {path} ({len(rows)} paires)")


def search(args):
    index_path = Path(args.run) / "index.npz"
    with np.load(index_path, allow_pickle=False) as data:
        if str(data["fingerprint"]) != fingerprint(args.run):
            raise ValueError("L'index ne correspond plus au modèle/tokenizer. Relancez index.")
        model, tokenizer = load_bundle(args.run)
        if args.text:
            query = model.get_layer("text_encoder")(tokenizer(tf.constant([args.text])), training=False).numpy()
            scores = (query @ data["images"].T)[0]
        else:
            query = model.get_layer("image_encoder")(load_image(args.image)[None, ...], training=False).numpy()
            scores = (query @ data["texts"].T)[0]
        results = [{"score": float(scores[i]), "image_path": str(data["paths"][i]),
                    "caption": str(data["captions"][i])} for i in np.argsort(-scores)[:args.top_k]]
    print(json.dumps(results, ensure_ascii=False, indent=2))


def recall_at_k(scores, ks=(1, 5, 10)):
    """Une paire positive par ligne ; les ex aequo suivent l'ordre de la galerie."""
    ranking = np.argsort(-scores, axis=1, kind="stable")
    targets = np.arange(len(scores))[:, None]
    return {f"Recall@{k}": float(np.mean(np.any(ranking[:, :k] == targets, axis=1)))
            for k in ks if k <= len(scores)}


def evaluate(args):
    rows = read_rows(Path(args.run) / "test.csv", ".")
    # Refuse les chemins du test qui figureraient dans les partitions d'apprentissage.
    for split in ("train", "val"):
        other = read_rows(Path(args.run) / f"{split}.csv", ".")
        if {r["image_path"] for r in rows} & {r["image_path"] for r in other}:
            raise ValueError("Chevauchement entre test et train/validation.")
    model, tokenizer = load_bundle(args.run)
    images, texts = embed(model, tokenizer, rows, args.batch_size)
    scores = images @ texts.T
    metrics = {"gallery_size": len(rows), "image_to_text": recall_at_k(scores),
               "text_to_image": recall_at_k(scores.T)}
    (Path(args.run) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
