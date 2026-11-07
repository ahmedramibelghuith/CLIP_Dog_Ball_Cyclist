"""Couches sérialisables et architectures dérivées des notebooks."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@keras.utils.register_keras_serializable(package="clip_project")
class L2Normalize(layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=-1)


@keras.utils.register_keras_serializable(package="clip_project")
class MaskedMean(layers.Layer):
    def call(self, inputs):
        sequence, tokens = inputs
        mask = tf.cast(tokens != 0, sequence.dtype)[..., None]
        return tf.reduce_sum(sequence * mask, axis=1) / tf.maximum(tf.reduce_sum(mask, axis=1), 1.)


@keras.utils.register_keras_serializable(package="clip_project")
class PositionalEmbedding(layers.Layer):
    def __init__(self, vocab_size, sequence_length, embed_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size, self.sequence_length, self.embed_dim = vocab_size, sequence_length, embed_dim
        self.tokens = layers.Embedding(vocab_size, embed_dim)
        self.positions = layers.Embedding(sequence_length, embed_dim)

    def call(self, inputs):
        return self.tokens(inputs) + self.positions(tf.range(tf.shape(inputs)[-1]))

    def get_config(self):
        return {**super().get_config(), "vocab_size": self.vocab_size,
                "sequence_length": self.sequence_length, "embed_dim": self.embed_dim}


@keras.utils.register_keras_serializable(package="clip_project")
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim, self.num_heads, self.ff_dim = embed_dim, num_heads, ff_dim
        self.dropout_rate = dropout_rate
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.norm1, self.norm2 = layers.LayerNormalization(epsilon=1e-6), layers.LayerNormalization(epsilon=1e-6)
        self.drop1, self.drop2 = layers.Dropout(dropout_rate), layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        sequence, tokens = inputs
        mask = (tokens != 0)[:, None, :]
        attention = self.attention(sequence, sequence, attention_mask=mask, training=training)
        x = self.norm1(sequence + self.drop1(attention, training=training))
        return self.norm2(x + self.drop2(self.ffn(x), training=training))

    def get_config(self):
        return {**super().get_config(), "embed_dim": self.embed_dim, "num_heads": self.num_heads,
                "ff_dim": self.ff_dim, "dropout_rate": self.dropout_rate}


@keras.utils.register_keras_serializable(package="clip_project")
class ClipLoss(layers.Layer):
    def __init__(self, temperature=0.07, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, inputs):
        image, text = inputs
        logits = tf.matmul(image, text, transpose_b=True) / self.temperature
        targets = tf.range(tf.shape(logits)[0])
        loss = keras.losses.sparse_categorical_crossentropy
        self.add_loss(tf.reduce_mean(loss(targets, logits, from_logits=True) +
                                     loss(targets, tf.transpose(logits), from_logits=True)) / 2.)
        return image, text

    def get_config(self):
        return {**super().get_config(), "temperature": self.temperature}


def image_classifier(num_classes):
    inputs = keras.Input((224, 224, 3), name="image_input")
    x = inputs
    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(.2)(x)
    return keras.Model(inputs, layers.Dense(num_classes, activation="softmax")(x))


def text_classifier(vocab_size, sequence_length, num_classes):
    tokens = keras.Input((sequence_length,), dtype="int32", name="text_input")
    x = PositionalEmbedding(vocab_size, sequence_length)(tokens)
    for _ in range(2):
        x = TransformerBlock()([x, tokens])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(.1)(x)
    x = MaskedMean(name="masked_mean")([x, tokens])
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(.2)(x)
    return keras.Model(tokens, layers.Dense(num_classes, activation="softmax")(x))


def clip_model(image_clf, text_clf, temperature=.07):
    image = L2Normalize()(layers.Dense(128)(image_clf.get_layer("flatten").output))
    text = L2Normalize()(layers.Dense(128)(text_clf.get_layer("masked_mean").output))
    image_encoder = keras.Model(image_clf.input, image, name="image_encoder")
    text_encoder = keras.Model(text_clf.input, text, name="text_encoder")
    img_in = keras.Input((224, 224, 3), name="image_input")
    txt_in = keras.Input(text_clf.input_shape[1:], dtype="int32", name="text_input")
    outputs = ClipLoss(temperature)([image_encoder(img_in), text_encoder(txt_in)])
    return keras.Model({"image_input": img_in, "text_input": txt_in}, outputs, name="clip_model")
