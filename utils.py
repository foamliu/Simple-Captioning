import pickle

import keras.backend as K
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.image import (load_img, img_to_array)


def predict(filename):
    img_rows, img_cols, img_size = 224, 224, 224
    max_token_length = 40
    start_word = '<start>'
    stop_word = '<end>'

    vocab = pickle.load(open('data/vocab_train.p', 'rb'))
    idx2word = sorted(vocab)
    word2idx = dict(zip(idx2word, range(len(vocab))))

    image_model = ResNet50(include_top=False, weights='imagenet', pooling='avg')
    img = load_img(filename, target_size=(img_rows, img_cols))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    image_input = np.empty((1, img_rows, img_cols, 3))
    image_input[0] = img_array
    encoding = image_model.predict(image_input)

    image_encoding = np.zeros((1, 2048))
    image_encoding[0] = encoding[0]

    model_path = 'models/model.04-1.3820-min.hdf5'
    model = load_model(model_path)
    print(model.summary())

    start_words = [start_word]
    while True:
        text_input = [word2idx[i] for i in start_words]
        text_input = sequence.pad_sequences([text_input], maxlen=max_token_length, padding='post')
        preds = model.predict([image_encoding, text_input])
        word_pred = idx2word[np.argmax(preds[0])]
        start_words.append(word_pred)
        if word_pred == stop_word or len(start_word) > max_token_length:
            break

    sentence = ' '.join(start_words[1:-1])
    print(sentence)

    K.clear_session()
