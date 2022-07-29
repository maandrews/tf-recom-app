"""
Customer related classes
"""
import tensorflow as tf
id_embed_dim = 8


class CustomerModel(tf.keras.Model):
    def __init__(self, unique_ids):
        super().__init__()

        self.id_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=unique_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_ids)+1, id_embed_dim, embeddings_initializer='uniform')
        ])

    def call(self, inputs):
        return tf.concat([
            self.id_embedding(inputs['USER_ID']),
        ], axis=1)


class QueryModel(tf.keras.Model):
    def __init__(self, layer_sizes, unique_uids):
        super().__init__()

        self.embedding_model = CustomerModel(unique_uids)

        self.dense_layers = tf.keras.Sequential()
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation='relu'))

        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)
