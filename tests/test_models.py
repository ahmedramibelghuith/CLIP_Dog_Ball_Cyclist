import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow non installé")
class ModelTests(unittest.TestCase):
    def test_normalization_and_serialization(self):
        import numpy as np
        import tensorflow as tf
        from clip_project.models import text_classifier, image_classifier, clip_model
        from clip_project.pipeline import load_image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = tf.fill((8, 8, 3), tf.constant(255, tf.uint8))
            path = root / "white.jpg"
            tf.io.write_file(str(path), tf.io.encode_jpeg(image))
            normalized = load_image(str(path))
            self.assertAlmostEqual(float(tf.reduce_max(normalized)), 1., places=5)
            model = clip_model(image_classifier(2), text_classifier(20, 8, 2))
            inputs = {"image_input": normalized[None, ...],
                      "text_input": tf.constant([[2, 3, 4, 0, 0, 0, 0, 0]])}
            expected = model(inputs, training=False)
            model.save(root / "clip.keras")
            restored = tf.keras.models.load_model(root / "clip.keras", compile=False)
            actual = restored(inputs, training=False)
            for before, after in zip(expected, actual):
                np.testing.assert_allclose(before.numpy(), after.numpy(), atol=1e-5)
                np.testing.assert_allclose(np.linalg.norm(after.numpy(), axis=1), 1., atol=1e-5)

    def test_tokenizer_roundtrip_and_recall(self):
        import numpy as np
        from tensorflow import keras
        from clip_project.retrieval import recall_at_k
        first = keras.layers.TextVectorization(output_sequence_length=8)
        first.adapt(["a dog runs", "a ball rolls"])
        second = keras.layers.TextVectorization(output_sequence_length=8)
        second.set_vocabulary(first.get_vocabulary())
        np.testing.assert_array_equal(first(["A DOG!"]).numpy(), second(["A DOG!"]).numpy())
        self.assertEqual(recall_at_k(np.eye(12))["Recall@1"], 1.)
        self.assertEqual(recall_at_k(np.roll(np.eye(12), 1, axis=1))["Recall@1"], 0.)
