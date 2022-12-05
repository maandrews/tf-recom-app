"""
File to train and save model
"""
import logging
import time
import pickle
import tensorflow as tf
import tensorflow_recommenders as tfrs
import pandas as pd
import numpy as np

from recommendation_model import RecomModel


logging.basicConfig(filename='create_data.log', encoding='utf-8',
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG, filemode='w')


timestr = time.strftime("%Y%m%d-%H%M%S")
tf.random.set_seed(12)
val_size = 1

df = pd.read_csv('../data/training_data.csv')

# Create training and validation (if we want, obviously this data is just random, and we won't train anything decent)
train = df.iloc[val_size:]
val = df.head(val_size)

unique_id_df = train[['USER_ID']].drop_duplicates()
unique_item_df = train[['ITEM']].drop_duplicates()

train_ds = tf.data.Dataset.from_tensor_slices(train.to_dict('list')).batch(1024)
val_ds = tf.data.Dataset.from_tensor_slices(val.to_dict('list')).batch(256)

ds_item = tf.data.Dataset.from_tensor_slices(unique_item_df.to_dict('list')).batch(256)
ds_id = tf.data.Dataset.from_tensor_slices(unique_id_df.to_dict('list')).batch(1024)

unique_items_for_metrics = ds_item.map(lambda x: {
    'ITEM': x['ITEM'],
})

unique_ids_for_metrics = ds_id.map(lambda x: {
    'USER_ID': x['USER_ID'],
})

train_item_buys = train_ds.map(lambda x: {
    'USER_ID': x['USER_ID'],
    'ITEM': x['ITEM'],
})

val_item_buys = val_ds.map(lambda x: {
    'USER_ID': x['USER_ID'],
    'ITEM': x['ITEM'],
})

items = ds_item.map(lambda x: x['ITEM'])
ids = ds_id.map(lambda x: x['USER_ID'])

unique_items = np.unique(np.concatenate(list(items)))
unique_ids = np.unique(np.concatenate(list(ids)))

unique_ids = unique_ids.astype(str)
unique_items = unique_items.astype(str)


if __name__ == "__main__":
    logging.info("Running train_model.py...")
    model = RecomModel(unique_ids, unique_items, unique_items_for_metrics, [32])
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    cached_train = train_ds.shuffle(1024).cache()
    cached_val = val_ds.shuffle(256).cache()

    training_history = model.fit(
        cached_train,
        validation_data = cached_val,
        validation_freq = 100,
        epochs = 10,
        verbose = 1
    )

    index = tfrs.layers.factorized_top_k.BruteForce(model.query_model, k=5)
    index.index_from_dataset(unique_items_for_metrics.map(model.candidate_model))

    query = {'USER_ID': tf.constant(['abc123'])}

    _, items = index(query)
    print(f"Recommendations for user: {items[0, :]}")

    # Save models
    model.query_model.save('../models/query_model_'+timestr)
    model.candidate_model.save('../models/candidate_model_'+timestr)
    weight_values = model.optimizer.get_weights()
    with open('../models/optimizer_state.pkl', 'wb') as f:
        pickle.dump(weight_values, f)

    # Save the index.
    path = '../models/index_'+timestr
    index.save(path, include_optimizer=False)

    # Load index.
    # loaded = tf.keras.models.load_model(path)

    # Recommendations using existing model:
    # path = '../models/index_0.4'
    # loaded_ix = tf.keras.models.load_model(path)
    #
    # query = {'cust_no': tf.constant(['104711']), 'gender': tf.constant([2.0])}
    # scores, titles = loaded_ix(query)
    #
    # # exclusions = tf.constant([['349462', '449584', '409993', '409993', '409993']])
    # # scores, titles = loaded_ix.query_with_exclusions(query, exclusions)
    #
    # print(f"Recommendations: {titles[0][:]}")
    # exit()
