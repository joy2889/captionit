from flask import Flask, render_template, request
import cv2
from keras.models import load_model
import numpy as np
from keras.applications import ResNet50
from keras.applications import ResNet101
from keras.applications import ResNet152
from keras.applications import MobileNet
from keras.applications import NASNetMobile
from keras.applications import Xception
from keras.preprocessing.sequence import pad_sequences
import os

#-------------RESNET50 VOCAB-------------------------------
vocab = np.load(r'Resnet50/vocab50.npy', allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v:k for k,v in vocab.items()}


#-------------RESNET101 VOCAB---------------------------------
vocab2 = np.load(r'Resnet101/vocab101.npy', allow_pickle=True)

vocab2 = vocab2.item()

inv_vocab2 = {v:k for k,v in vocab2.items()}


#-------------------------------------------------------------
vocab3 = np.load(r'mob/vocabmob.npy', allow_pickle=True)

vocab3 = vocab3.item()

inv_vocab3 = {v:k for k,v in vocab3.items()}

#-------------------------------------------------------------

vocab4 = np.load(r'nas/vocabnas.npy', allow_pickle=True)

vocab4 = vocab4.item()

inv_vocab4 = {v:k for k,v in vocab4.items()}

#-------------------------------------------------------------

vocab5 = np.load(r'xc/vocabxc.npy', allow_pickle=True)

vocab5 = vocab5.item()

inv_vocab5 = {v:k for k,v in vocab5.items()}

print("+"*50)
print("vocabulary loaded")


#--------------------RESNET50 LSTM MODEL---------------------
embedding_size = 128
vocab_size = len(vocab)
max_len = 40

model = load_model(r'Resnet50/model50.h5')

model.load_weights(r'Resnet50/mine_model_weights50.h5')


#--------------------RESNET101 LSTM MODEL--------------------
max_len = 35

model2 = load_model(r'Resnet101/model101.h5')

model2.load_weights(r'Resnet101/mine_model_weights101.h5')


#------------------------------------------------------------

model3 = load_model(r'Resnet152/model152.h5')

model3.load_weights(r'Resnet152/mine_model_weights152.h5')

#------------------------------------------------------------

model4 = load_model(r'mob/modelmob.h5')

model4.load_weights(r'mob/mine_model_weightsmob.h5')

#------------------------------------------------------------

model5 = load_model(r'nas/modelnas.h5')

model5.load_weights(r'nas/mine_model_weightsnas.h5')

#------------------------------------------------------------

model6 = load_model(r'xc/modelxc.h5')

model6.load_weights(r'xc/mine_model_weightsxc.h5')


print("="*150)
print("MODEL LOADED")

resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')

resnet2 = ResNet101(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')

resnet3 = ResNet152(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')

mob_obj = MobileNet(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')

nas_obj = NASNetMobile(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')

xc_obj = Xception(include_top=False,weights='imagenet',input_shape=(299,299,3),pooling='avg')


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

    global model, resnet, vocab, inv_vocab, model2, vocab2, resnet2, inv_vocab2, vocab3, inv_vocab3, mob_obj

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
    incept2 = resnet2.predict(image).reshape(1,2048)
    incept3 = resnet3.predict(image).reshape(1,2048)
    incept4 = mob_obj.predict(image).reshape(1,1024)
    incept5 = nas_obj.predict(image).reshape(1,1056)
    
    
    image2 = cv2.imread(path)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2 = cv2.resize(image2, (299,299))
    image2 = np.reshape(image2, (1,299,299,3))
    incept6 = xc_obj.predict(image2).reshape(1,2048)
    
    print("="*50)
    print("Predict Features")


    text_in = ['startofseq']

    final = ''
    final2 = ''
    final3 = ''
    final4 = ''
    final5 = ''
    final6 = ''

    print("="*50)
    print("GETING Captions")
#-------------------------------------------------------------------------------------------------
    count = 0
    while count < 20:

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab[i])

        padded = pad_sequences([encoded], maxlen=40, padding='post', truncating='post').reshape(1,40)

        sampled_index = np.argmax(model.predict([incept, padded]))

        sampled_word = inv_vocab[sampled_index]

        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word

        text_in.append(sampled_word)
#-------------------------------------------------------------------------------------------------
    
    count = 0
    text_in = ['startofseq']
    while count < 20:

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab2[i])

        padded = pad_sequences([encoded], maxlen=35, padding='post', truncating='post').reshape(1,35)

        sampled_index = np.argmax(model2.predict([incept2, padded]))

        sampled_word = inv_vocab2[sampled_index]

        if sampled_word != 'endofseq':
            final2 = final2 + ' ' + sampled_word

        text_in.append(sampled_word)

#-------------------------------------------------------------------------------------------------

    count = 0
    text_in = ['startofseq']
    while count < 20:

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab2[i])

        padded = pad_sequences([encoded], maxlen=35, padding='post', truncating='post').reshape(1,35)

        sampled_index = np.argmax(model3.predict([incept3, padded]))

        sampled_word = inv_vocab2[sampled_index]

        if sampled_word != 'endofseq':
            final3 = final3 + ' ' + sampled_word

        text_in.append(sampled_word)

#--------------------------------------------------------------------------------------------------

    count = 0
    text_in = ['startofseq']
    while count < 20:

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab3[i])

        padded = pad_sequences([encoded], maxlen=40, padding='post', truncating='post').reshape(1,40)

        sampled_index = np.argmax(model4.predict([incept4, padded]))

        sampled_word = inv_vocab3[sampled_index]

        if sampled_word != 'endofseq':
            final4 = final4 + ' ' + sampled_word

        text_in.append(sampled_word)
        
#--------------------------------------------------------------------------------------------------

    count = 0
    text_in = ['startofseq']
    while count < 20:

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab4[i])

        padded = pad_sequences([encoded], maxlen=40, padding='post', truncating='post').reshape(1,40)

        sampled_index = np.argmax(model5.predict([incept5, padded]))

        sampled_word = inv_vocab4[sampled_index]

        if sampled_word != 'endofseq':
            final5 = final5 + ' ' + sampled_word

        text_in.append(sampled_word)
        
#--------------------------------------------------------------------------------------------------

    count = 0
    text_in = ['startofseq']
    while count < 20:

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab5[i])

        padded = pad_sequences([encoded], maxlen=40, padding='post', truncating='post').reshape(1,40)

        sampled_index = np.argmax(model6.predict([incept6, padded]))

        sampled_word = inv_vocab5[sampled_index]

        if sampled_word != 'endofseq':
            final6 = final6 + ' ' + sampled_word

        text_in.append(sampled_word)



    return render_template('after.html', data=final,p = path,l=final2,r=final3,j=final4,k=final5,m=final6)

if __name__ == "__main__":
    app.run(host = "0.0.0.0")


