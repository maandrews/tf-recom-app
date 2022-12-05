"""
Web app to input user IDs and receive recommendations.
"""

from flask import Flask
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_recommenders as tfrs

path = '../models/index_20220725-172021'
index = tf.keras.models.load_model(path)

app = Flask(__name__)




if __name__ == '__main__':
    app.run()