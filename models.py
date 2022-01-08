
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from tensorflow.keras.layers import Conv2D
#from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import LSTM, GRU, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing import text, sequence
#from tensorflow.compat.v1.keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
#from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
#from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
#from keras.engine.topology import Layer
from tensorflow.keras.layers import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


from keras.layers import *
from keras.models import *
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.initializers import *
from keras.optimizers import *
import keras.backend as K
from keras.callbacks import *
import tensorflow as tf
import os
import time
import gc
import glob

def LSTM_model(embedding_matrix,vocab_len,emb_dim):
    lstm_model = Sequential()
    lstm_model.add(Embedding(vocab_len, emb_dim, trainable=False, weights=[embedding_matrix]))
    lstm_model.add(LSTM(128, return_sequences=False))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(Dense(6, activation='sigmoid'))
    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(lstm_model.summary())
    return lstm_model

def CNN_model(embedding_matrix):
    emb_dim = embedding_matrix.shape[1]
    cnn_model = Sequential()
    cnn_model.add(Embedding(vocab_len, emb_dim, trainable=False, weights=[embedding_matrix]))
    cnn_model.add(layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    cnn_model.add(layers.MaxPooling1D(5))
    cnn_model.add(layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
    cnn_model.add(layers.GlobalMaxPooling1D())
    cnn_model.add(layers.Dense(6, activation='sigmoid'))
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model.summary()
    return cnn_model

def GRU_model(embedding_matrix):
    emb_dim = embedding_matrix.shape[1]
    gru_model = Sequential()
    gru_model.add(Embedding(vocab_len, emb_dim, trainable=False, weights=[embedding_matrix]))
    gru_model.add(GRU(128, return_sequences=False))
    gru_model.add(Dropout(0.5))
    gru_model.add(Dense(6, activation='sigmoid'))
    gru_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    gru_model.summary()
    return(gru_model)

def CNN_GRU_model(embedding_matrix):
    emb_dim = embedding_matrix.shape[1]
    cnn_gru_model = Sequential()
    cnn_gru_model.add(Embedding(vocab_len, emb_dim, trainable=False, weights=[embedding_matrix]))
    cnn_gru_model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
    cnn_gru_model.add(MaxPooling1D(pool_size=2))
    # cnn_gru_model.add(Dropout(0.25))
    cnn_gru_model.add(GRU(128, return_sequences=True))
    # cnn_gru_model.add(Dropout(0.3))
    # cnn_gru_model.add(Flatten())
    cnn_gru_model.add(layers.GRU(128))
    # cnn_gru_model.add(Dense(128,activation='relu'))
    # cnn_gru_model.add(Dropout(0.5))
    cnn_gru_model.add(Dense(6, activation='softmax'))
    cnn_gru_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    cnn_gru_model.summary()
    return(cnn_gru_model)

def CNN_LSTM_model(embedding_matrix):
    emb_dim = embedding_matrix.shape[1]
    cnn_lstm_model = Sequential()
    cnn_lstm_model.add(Embedding(vocab_len, emb_dim, trainable=False, weights=[embedding_matrix]))
    cnn_lstm_model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
    cnn_lstm_model.add(MaxPooling1D(pool_size=2))
    # cnn_gru_model.add(Dropout(0.25))
    cnn_lstm_model.add(LSTM(128, return_sequences=True))
    # cnn_gru_model.add(Dropout(0.3))
    # cnn_gru_model.add(Flatten())
    cnn_lstm_model.add(layers.LSTM(128))
    # cnn_gru_model.add(Dense(128,activation='relu'))
    # cnn_gru_model.add(Dropout(0.5))
    cnn_lstm_model.add(Dense(6, activation='softmax'))
    cnn_lstm_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    cnn_lstm_model.summary()
def BID_LSTM_model(embedding_matrix):
    lstmbd_model = Sequential()
    lstmbd_model.add(Embedding(vocab_len, emb_dim, trainable=False, weights=[embedding_matrix]))
    # lstmbd_model.add(LSTM(128, return_sequences=False))
    lstmbd_model.add(Bidirectional(LSTM(128, return_sequences=False)))
    lstmbd_model.add(Dropout(0.5))
    lstmbd_model.add(Dense(6, activation='softmax'))
    lstmbd_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstmbd_model.summary()
    return(lstmbd_model)

def BID_GRU_model(embedding_matrix):
    grubd_model = Sequential()
    grubd_model.add(Embedding(vocab_len, emb_dim, trainable=False, weights=[embedding_matrix]))
    # grubd_model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1],weights=[embedding_matrix],trainable=True))
    # grubd_modeladd(Embedding(max_features, 100, input_length=500))
    # 128, return_sequences=False))
    # grubd_model.add(SpatialDropout1D(0.25))
    grubd_model.add(Bidirectional(GRU(128)))
    grubd_model.add(Dropout(0.5))

    grubd_model.add(Dense(6, activation='softmax'))
    grubd_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    grubd_model.summary()
    return(grubd_model)
