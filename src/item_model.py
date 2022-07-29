"""
Item related models
"""
import tensorflow as tf
item_embed_dim = 8


class ItemModel(tf.keras.Model):
    def __init__(self, unique_items):
        super().__init__()

        self.item_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=unique_items, mask_token=None),
            tf.keras.layers.Embedding(len(unique_items) + 1, item_embed_dim, embeddings_initializer='uniform')
        ])

    def call(self, inputs):
        return tf.concat([
            self.item_embedding(inputs['ITEM']),
        ], axis=1)


class CandidateModel(tf.keras.Model):
    def __init__(self, layer_sizes, unique_items):
        super().__init__()

        self.embedding_model = ItemModel(unique_items)

        self.dense_layers = tf.keras.Sequential()
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation='relu'))

        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)
