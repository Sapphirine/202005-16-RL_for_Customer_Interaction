import numpy as np
import pandas as pd
import datetime
from calendar import monthrange
from tqdm import tqdm
from sklearn import preprocessing
import warnings
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
import tensorflow as tf
import sklearn
from tensorflow import keras
from tensorflow.keras import layers
from keras import callbacks
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from keras import Sequential
from keras.layers import LSTM, Dense
from keras import backend as K
from keras.models import load_model


def data_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def get_lstm(x_tr, y_tr, x_val, y_val, lstm_epochs, lstm_batch_size,episode_length):
    K.clear_session()

    model = Sequential()
    model.add(LSTM(episode_length, go_backwards=True))
    model.add(Dense(episode_length))

    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mse'])

    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       restore_best_weights=True,
                       verbose=1,
                       patience=50)

    mc = callbacks.ModelCheckpoint('best_model.h5',
                                   monitor='val_loss',
                                   mode='min',
                                   save_best_only=True,
                                   verbose=1,
                                   save_weights_only=True)

    model.fit(x_tr, y_tr,
              validation_data=[x_val, y_val],
              callbacks=[es, mc],
              epochs=lstm_epochs,
              batch_size=lstm_batch_size,
              verbose=1,
              shuffle=True)

    model.load_weights("best_model.h5")

    return model


def base(states, v,base_epochs,base_batch_size):
    X_train, X_test, y_train, y_test = train_test_split(states, v, test_size=0.2, random_state=42)

    K.clear_session()

    model = load_model('my_model.hdf5')

    es = EarlyStopping(monitor='val_loss',
                       mode='min',
                       restore_best_weights=True,
                       verbose=1,
                       patience=50)

    mc = callbacks.ModelCheckpoint('best_model.h5',
                                   monitor='val_loss',
                                   mode='min',
                                   save_best_only=True,
                                   verbose=1,
                                   save_weights_only=True)

    model.fit(X_train, y_train,
              validation_data=[X_test, y_test],
              callbacks=[es, mc],
              epochs=base_epochs,
              batch_size=base_batch_size,
              verbose=1,
              shuffle=True)

    model.load_weights("best_model.h5")

    return model



def direct_rl(model, states, rewards, alpha, gamma,base_epochs,base_batch_size,episode,episode_length,iteration):        # here rewards are estimated cumulative rewards
  Q = []
  v = model.predict(states)        #return the estimated cumulative rewards for each states and actions
  for k in range(iteration):
    cnt = 0
    for i in range(episode):
      for j in range(episode_length):
        if j != (episode_length-1):
          v[i][j] = v[i][j] + alpha*(rewards[cnt]+gamma*v[i][j+1]-v[i][j])
        cnt += 1
    model = base(states, v,base_epochs,base_batch_size)
    v = model.predict(states)
    Q.append(v)
  return Q, model




def indirect_rl(model, states, rewards,alpha,gamma,base_epochs,base_batch_size,episode,episode_length,iteration):        # here rewards are estimated cumulative rewards
  Q = []
  v = model.predict(states)        #return the estimated cumulative rewards for each states and actions
  for k in range(iteration):
    epi = Simulate(states, model)
    print('success')
    cnt = 0
    for i in range(episode):
      for j in range(episode_length):
        if j != episode_length-1:
          v[i][j] = v[i][j] + alpha*(rewards[cnt]+gamma*v[i][j+1]-v[i][j])
        cnt += 1
    model = base(epi,v,base_epochs,base_batch_size)
    v = model.predict(epi)
    Q.append(v)
  return Q,model



def Simulate(states, model,alpha, gamma):
    X = states.copy()
    X[:, :, -1] = 0
    y1 = model.predict(X)[0]
    X[:, :, -1] = 1
    y2 = model.predict(X)[0]
    X[:, :, -1] = ((y2 - y1) > 0) * 1
    return X



def semidirect_rl(model,states,rewards,alpha,gamma,base_epochs,base_batch_size,episode,episode_length,iteration):        # here rewards are estimated cumulative rewards
    rewards=np.array_split(rewards,episode)
    X_sampled, X_other, rewards_sampled, y_other = train_test_split(states, rewards, test_size=0.5, random_state=1)
    Q = []
    v = model.predict(states)        #return the estimated cumulative rewards for each states and actions
    for k in range(iteration):
        X_sampled, X_other, y_sampled, y_other = train_test_split(states, rewards, test_size=0.5, random_state=1)
        temp=X_sampled.copy()
        temp[:,:,-1]=1-X_sampled[:,:,-1]
        v=model.predict(X_sampled)
        y2=model.predict(temp)
        p=((v-y2)>0)*1
    for i in range(len(X_sampled)):
        for j in range(episode_length):
            if j != (episode_length-1) and p[i][j] == 1 :
                v[i][j] = v[i][j] + alpha*(rewards_sampled[i][j]+gamma*v[i][j+1]-v[i][j])
    model = base(X_sampled, v,base_epochs,base_batch_size)
    v = model.predict(states)
    Q.append(v)
    return Q,model


