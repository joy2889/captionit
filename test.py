# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 08:17:36 2021

@author: joyme
"""

from flask import Flask, render_template, request
import cv2
from keras.models import load_model
import numpy as np
from keras.applications import ResNet101
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
import cv2
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import os

vocab = np.load('vocab101.npy', allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v:k for k,v in vocab.items()}


print("+"*50)
print("vocabulary loaded")

embedding_size = 128
vocab_size = len(vocab)
max_len = 35

model = load_model('model101.h5')

model.load_weights('mine_model_weights101.h5')

print("="*150)
print("MODEL LOADED")

resnet = ResNet101(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')




print("="*150)
print("RESNET MODEL LOADED")




app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template(r'solid/index.html')

@app.route('/next')
def next():
    return render_template("next.html")

@app.route('/after', methods=['GET', 'POST'])

def after():

    global model, resnet, vocab, inv_vocab

    img = request.files['file1']

    
    path = os.path.join(r'./static', img.filename)

    img.save(path)

    print("="*50)
    print("IMAGE SAVED")


    
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224,224))

    image = np.reshape(image, (1,224,224,3))

    
    
    incept = resnet.predict(image).reshape(1,2048)

    print("="*50)
    print("Predict Features")


    text_in = ['startofseq']

    final = ''

    print("="*50)
    print("GETING Captions")

    count = 0
    while tqdm(count < 20):

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab[i])

        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1,max_len)

        sampled_index = np.argmax(model.predict([incept, padded]))

        sampled_word = inv_vocab[sampled_index]

        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word

        text_in.append(sampled_word)


    return render_template('after.html', data=final,p = path)

if __name__ == "__main__":
    app.run(host = "127.0.0.1", port = 5000, debug=True)


