from __future__ import print_function, division
import scipy
from keras import metrics
from keras.layers import Input, Dense, Embedding, LSTM, Reshape, Softmax
from keras.models import Model, load_model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import subfunc_1 as subs
import tensorflow as tf



class Universal():
    def __init__(self):
        # Input shape
        self.UNIFAC_shape = (None, 119, 26) #(length, group)
        self.smile_shape = (None, 50)  #(length, char)
        self.sigma_shape = (None, 51)  #(length, char)
        self.x_shape = (None, 50, 200)
        # Load data
        self.TOPO_data = self.load_value_data('./DataPool/TOPO2048.csv')
        self.MACCS_data = self.load_value_data('./DataPool/MACCS.csv')
        self.smile_data = self.load_value_data('./DataPool/SMILES.csv')
        self.Morgan_data = self.load_value_data('./DataPool/Morgan512.csv')
        self.compound_data = self.load_compound_data('./DataPool/compounds.csv')

        #set optimizer
        optimizer = Adam(0.001, 0.9)   # (lr,beta)

        # Build the Encoder
        self.E_S2X = self.build_E_S2X()

        # Build and compile for Decoder
        self.D_X2MO = self.build_D_X2MO()
        self.D_X2MA = self.build_D_X2MA()
        self.D_X2T = self.build_D_X2T()

        # Input images from both domains
        smiles = Input(batch_shape=self.smile_shape)

        # Translate images to the other domain
        x_from_s = self.E_S2X(smiles)

        # Translate images back to original domain
        pred_MO = self.D_X2MO(x_from_s)
        pred_MA = self.D_X2MA(x_from_s)
        pred_T = self.D_X2T(x_from_s)

        # Combined model trains generators to fool discriminators
        self.AE_combined = Model(inputs=smiles,
                                 outputs=[pred_MO, pred_MA, pred_T])
        self.AE_combined.compile(loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
                                 loss_weights=[4, 10, 1],
                                 optimizer=optimizer,
                                 metrics=[subs.r2, metrics.binary_accuracy]
                                 )
        self.embedding = Model(inputs=smiles,
                               outputs=x_from_s)
        self.AE_combined.summary()
        # self.save_generator()

    def load_value_data(self, path):
        data = pd.read_csv(path, header=None, index_col=None).values
        return data

    def load_compound_data(self, path):
        c = list(pd.read_csv(path, header=None, index_col=None)[0])
        return c

    def build_E_S2X(self):
        # embedding model
        input_shape = Input(batch_shape=self.smile_shape)
        embedding_layer = Embedding(input_dim=182, input_length=50, output_dim=200, mask_zero=True)(input_shape)
        return Model(input_shape, embedding_layer, name='E_S2X')

    def build_D_X2MO(self):
        # Morgan predictor
        z = Input(batch_shape=self.x_shape)
        L0 = LSTM(200, return_sequences=False, return_state=False, activation='tanh')(z)
        calculator = Dense(512, activation='sigmoid', name='Morgan')(L0)
        return Model(z, calculator, name='D_X2MO')

    def build_D_X2MA(self):
        # MACCS predictor
        z = Input(batch_shape=self.x_shape)
        L0 = LSTM(200, return_sequences=False, return_state=False, activation='tanh')(z)
        calculator = Dense(166, activation='sigmoid', name='MACCS')(L0)
        return Model(z, calculator, name='D_X2M')

    def build_D_X2T(self):
        # TOPO predictor
        z = Input(batch_shape=self.x_shape)
        L0 = LSTM(200, return_sequences=False, return_state=False, activation='tanh')(z)
        calculator = Dense(2048, activation='sigmoid', name='TOPO')(L0)
        return Model(z, calculator, name='D_X2T')

    def build_D_X2S(self):
        # sigma profile predictor
        z = Input(batch_shape=self.x_shape)
        LSTM1 = LSTM(units=1000, return_sequences=False, stateful=False, activation='tanh', dropout=0.5,
                     recurrent_dropout=0.5)(z)
        Dense1 = Dense(units=1000, activation='relu')(LSTM1)
        Dense2 = Dense(units=1000, activation='relu')(Dense1)
        Dense3 = Dense(units=1000, activation='relu')(Dense2)
        Dense4 = Dense(units=1000, activation='relu')(Dense3)
        sigma = Dense(units=51, activation='relu', name='sigma')(Dense4)
        return Model(input=z, output=sigma, name='D_X2S')

    def train(self, epochs, batch_size=1):

        smile_train, smile_test = train_test_split(self.smile_data, test_size=0.2, random_state=33)
        Morgan_train, Morgan_test = train_test_split(self.Morgan_data, test_size=0.2, random_state=33)
        maccs_train, maccs_test = train_test_split(self.MACCS_data, test_size=0.2, random_state=33)
        TOPO_train, TOPO_test = train_test_split(self.TOPO_data, test_size=0.2, random_state=33)

        EarlyStopping_callback = EarlyStopping(patience=100, monitor='val_loss', mode='min')
        CSVLogger_callback = CSVLogger('./logs.csv')
        weight_save_callback = ModelCheckpoint('./model_save/MA & MO & T_E500_100_n.h5',
                                               monitor='val_loss', verbose=0,
                                               save_best_only=True, mode='min', period=1)

        '''
        self.AE_combined = Model(inputs=smile,
                                 outputs=[reconstr_u, generate_sigma])
        '''

        g_loss = self.AE_combined.fit(smile_train, [Morgan_train, maccs_train, TOPO_train],
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      validation_data=(smile_test[:64],
                                                       [Morgan_test[:64], maccs_test[:64], TOPO_test[:64]]),
                                      callbacks=[CSVLogger_callback, weight_save_callback, EarlyStopping_callback])

        self.embedding.save('./model_save/embedding500_MA & MO & T200_200_nn.h5')
        self.AE_combined.save('./model_save/MA & MO & T_E200_200_nn.h5')

    def load_model(self):
        self.AE_combined = load_model('./model_save/model_save/MA & MO & T.h5')

    def test(self, R2_bound = 0.8):
        smile_train, smile_test = train_test_split(self.smile_data, test_size=0.2, random_state=33)
        Morgan_train, Morgan_test = train_test_split(self.Morgan_data, test_size=0.2, random_state=33)
        maccs_train, maccs_test = train_test_split(self.MACCS_data, test_size=0.2, random_state=33)
        TOPO_train, TOPO_test = train_test_split(self.TOPO_data, test_size=0.2, random_state=33)

        self.model = load_model('./model_save/MA & MO & T_E500_100_nn.h5', custom_objects={'r2': subs.r2,
                                                                                    'acc': metrics.binary_accuracy})
        #MO_pred, MA_pred, TOPO_pred = self.model.predict(smile_train)
        #MO_pred_t, MA_pred_t, TOPO_pred_t = self.model.predict(smile_test)

        loss = self.model.evaluate(x=smile_train, y=[Morgan_train, maccs_train, TOPO_train], batch_size=512)
        loss_t = self.model.evaluate(x=smile_test, y=[Morgan_test, maccs_test, TOPO_test], batch_size=512)
        print(loss, loss_t)
        return loss, loss_t


# 如果在其他的py檔想執行universal 那以下單元測試也會被執行 可以把if判斷改成false就不會執行(how?)
if __name__ == '__main__':
    uni = Universal()  # load data build model
    #uni.train(epochs=5000, batch_size=512)
    uni.test()
    # uni.S2M_test()
    # uni.S2S_test()

