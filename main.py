import json
import os
import cv2 as cv
import numpy as np
import librosa
import librosa.display
from pathlib import Path
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras import metrics, optimizers
from werkzeug.utils import secure_filename
import pickle

from flask import Flask, request, jsonify

# define a flask app
app = Flask(__name__)

model = pickle.load(open('models/final_temperature_detection_model.pkl', 'rb'))

@app.route('/temperature',methods=['POST'])
def temperature():
    # get the data from the POST request.
    data = request.get_json(force=True)
    # make prediction using model loaded from disk as per the data.
    prediction = model.predict([np.array(data['data'])])
    # take the first value of prediction
    temp = prediction[0]
    return jsonify(temperature=json.dumps(temp.astype(float)))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
