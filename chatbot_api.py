import tensorflow as tf
from flask import Flask, request, jsonify, redirect, url_for
from flask import send_file
import datetime
import numpy as np
import pandas as pd
import pickle
import os
import json 
import csv
import requests
import time
from transformers import AutoTokenizer, AutoModel
import joblib
import shutil 


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

# Path: /api
@app.route('/api_dl', methods=['GET','POST'])
def api_dl():
    # try:
        message = str(request.form.get('message'))
        print(message)

        def convert_to_3D(data):
            return data.reshape((1,data.shape[0],1,1))
        
        word = message
        max_token_lenth = 256
        tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
        model = AutoModel.from_pretrained('vinai/phobert-base')
        encoded = tokenizer(word, return_tensors='pt', max_length=max_token_lenth, truncation=True,padding=True)
        embedding_vector = model(**encoded).pooler_output.detach().numpy()[0]
        # load model
        model = tf.keras.models.load_model(f'./models/20230711-165851-model.h5')

        embedding_vector = convert_to_3D(embedding_vector)
        # print(embedding_vector)
        print(embedding_vector.shape)
        result = model.predict([embedding_vector])
        print(result)
        print(np.argmax(result,axis=-1))
        json_result = {
            'result': str(np.argmax(result,axis=-1)[0]),
        }
        # [[]] to json
        json_confidence = {
            'confidence': result.tolist()[0],
            'result': str(np.argmax(result,axis=-1)[0]),
            'message': message,
            'time': str(datetime.datetime.now()),
            'model': '20230711-165851-model.h5',
            'tokenizer': 'vinai/phobert-base', 
            'embedding': 'vinai/phobert-base',
            'success': 'true'
        }
        print(json_confidence)
        return jsonify(json_confidence)
    
    # except Exception as e:
    #     return jsonify({'error': str(e)})


@app.route('/api_ml', methods=['GET','POST'])
def api_ml():
    # try:
        message = str(request.form.get('message'))
        print(message)
        max_token_lenth = 256
        
        word = message
        clf = joblib.load(f'./models/20230711-174848_model.sav')

        encoded_input = tokenizer(word, return_tensors='pt', max_length=max_token_lenth, truncation=True,padding=True)
        embedding_vector = model(**encoded_input).pooler_output.detach().numpy()[0]

        # clf.predict([embedding_vector])
        
        json_result = {
            'result': str(clf.predict([embedding_vector])[0]),
        }
        print(json_result)
        return jsonify(json_result)
    # except Exception as e:
        # return jsonify({'error': str(e)})
if __name__ == '__main__':
    port_sv = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port_sv)