"""Entraînement et export d'un ensemble modèle/tokenizer cohérent."""
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from .data import read_rows, split_rows, write_rows
from .models import image_classifier, text_classifier, clip_model


def load_image(path):
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.resize(image, (224, 224))


def dataset(rows, tokenizer, classes, batch_size, mode, training=False, seed=42):
    paths = [r["image_path"] for r in rows]
    texts = [r["caption"] for r in rows]
    labels = [classes.index(r["label"]) for r in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, texts, labels))
    if training:
        ds = ds.shuffle(len(rows), seed=seed, reshuffle_each_iteration=True)

    def convert(path, text, label):
        if mode == "image":
            return load_image(path), label
        tokens = tf.cast(tokenizer(text), tf.int32)
        if mode == "text":
            return tokens, label
        return {"image_input": load_image(path), "text_input": tokens}

    ds = ds.map(convert, num_parallel_calls=tf.data.AUTOTUNE)
    # Évite une perte contrastive nulle sur un dernier batch d'une seule paire.
    drop = mode == "clip" and training and len(rows) % batch_size == 1
    return ds.batch(batch_size, drop_remainder=drop).prefetch(tf.data.AUTOTUNE)


def fit(model, parts, tokenizer, classes, args, mode, epochs, learning_rate, out):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  **({"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"]}
                     if mode != "clip" else {}))
    history = model.fit(
        dataset(parts["train"], tokenizer, classes, args.batch_size, mode, True, args.seed),
        validation_data=dataset(parts["val"], tokenizer, classes, args.batch_size, mode),
        epochs=epochs,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
    model.save(out / f"{mode}.keras")
    (out / f"{mode}_history.json").write_text(json.dumps(history.history, indent=2))


def train(args):
    keras.utils.set_random_seed(args.seed)
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Le dossier de sortie doit être vide : choisissez un nouveau dossier de run.")
    rows = read_rows(args.csv, args.image_root)
    parts = split_rows(rows, args.seed)
    classes = sorted({r["label"] for r in rows})
    tokenizer = keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=args.sequence_length,
                                               standardize="lower_and_strip_punctuation")
    tokenizer.adapt(tf.constant([r["caption"] for r in parts["train"]]))
    out.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        write_rows(out / f"{name}.csv", part)
    config = {"seed": args.seed, "sequence_length": args.sequence_length, "image_size": [224, 224],
              "embed_dim": 128, "classes": classes, "batch_size": args.batch_size,
              "classifier_epochs": args.classifier_epochs, "clip_epochs": args.clip_epochs,
              "tensorflow_version": tf.__version__, "keras_version": keras.__version__,
              "standardize": "lower_and_strip_punctuation", "vocabulary": tokenizer.get_vocabulary()}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    image_clf = image_classifier(len(classes))
    fit(image_clf, parts, tokenizer, classes, args, "image", args.classifier_epochs, 1e-3, out)
    text_clf = text_classifier(len(config["vocabulary"]), args.sequence_length, len(classes))
    fit(text_clf, parts, tokenizer, classes, args, "text", args.classifier_epochs, 1e-4, out)
    model = clip_model(image_clf, text_clf)
    fit(model, parts, tokenizer, classes, args, "clip", args.clip_epochs, 1e-4, out)
    print(f"Run sauvegardé : {out}")


def load_bundle(run):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    tokenizer = keras.layers.TextVectorization(output_sequence_length=config["sequence_length"],
                                               standardize=config["standardize"])
    tokenizer.set_vocabulary(config["vocabulary"])  # conserve PAD et OOV
    model = keras.models.load_model(run / "clip.keras", compile=False)
    return model, tokenizer
