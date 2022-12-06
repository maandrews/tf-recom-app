"""
Web app to input user IDs and receive recommendations.
"""

from flask import Flask, render_template, request
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_recommenders as tfrs

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        userid = request.form['userid']
        query = {'USER_ID': tf.constant([str(userid)])}
        _, recoms = index(query)
        # print(f"Recommendations: {titles[0][:]}") s0hghhgp
        return render_template('index.html', pred=f"Recommendations: {recoms[0, :]}")

    return render_template('index.html')


if __name__ == '__main__':
    path = '../models/index_20220725-172021'
    index = tf.keras.models.load_model(path)
    app.run()
