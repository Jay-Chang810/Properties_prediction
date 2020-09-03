import numpy as np
from keras.layers import Dense, Input, LSTM, Embedding
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard
import subfunc_1 as subs
import pandas as pd
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

K.clear_session()
# load data
Propers = pd.read_csv('./DataPool/GDB_9.csv', index_col=None, header=None).values
compound= list(pd.read_csv('./DataPool/compounds.csv', index_col=None, header=None)[0])
smiles = pd.read_csv('./DataPool/smiles.csv', index_col=None, header=None).values


# data scaling between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
scaler.fit((Propers))
Propers1 = scaler.transform((Propers))

xtrain, xtest = train_test_split(smiles, test_size=0.7, random_state=10)
ytrain, ytest = train_test_split(Propers, test_size=0.7, random_state=10)
compound_train, compound_test = train_test_split(compound, test_size=0.7, random_state=10)

# choose embedding layer model and project smiles to x
output_dim = 500
K.clear_session()
load_model_name = './model_save/embedding171K_MA & MO & T500_200.h5'
embedding = load_model(load_model_name, custom_objects={'r2': subs.r2})
x_embedding = embedding.predict(xtrain)
x_embedding_test = embedding.predict(xtest)

model_name = './model_save/GDB_9.h5'

# build predict model

# Input
input_shape = Input(batch_shape=(None, 50, output_dim))
# hidden layer
LSTM1 = LSTM(units=1000, return_sequences=False, stateful=False, activation='tanh', dropout=0.5,
             recurrent_dropout=0.5)(input_shape)
Dense1 = Dense(units=1000, activation='relu')(LSTM1)
Dense1 = Dense(units=1000, activation='relu')(Dense1)
# output
calculator = Dense(units=3, activation='relu')(Dense1)
Dense_model = Model(input=input_shape, output=calculator)
Dense_model.summary()

# Callbacks
EarlyStopping_callback = EarlyStopping(patience=100, monitor='val_loss', mode='min')
weight_save_callback = ModelCheckpoint(model_name,
                                    monitor='val_loss', verbose=0, save_best_only=True, mode='min', period=1)

# model compile
Dense_model.compile(optimizer='Adam', loss='mae')
Dense_model.fit(x_embedding, ytrain,
                nb_epoch=5000,
                batch_size=512,
                validation_data=(x_embedding_test[:256], ytest[:256]),
                shuffle=True, callbacks=[weight_save_callback, EarlyStopping_callback,])

# save model
Dense_model.save(model_name)


load_model_name = './model_save/embedding_MA & MO & T500_200.h5'
model_name = './model_save/GDB_9.h5'
# test
embedding = load_model(load_model_name, custom_objects={'r2': subs.r2})
x_embedding = embedding.predict(xtrain)
x_embedding_test = embedding.predict(xtest)
Dense_model = load_model(model_name, custom_objects={'r2': subs.r2})
y_train_pred = Dense_model.predict(x_embedding)
y_test_pred = Dense_model.predict(x_embedding_test)
loss = Dense_model.evaluate(x=x_embedding_test, y=ytest, batch_size=2000)
print('\ntest loss:', loss)

def scale_back(x, y):
    # x 是原始值 y是被scale後的值
    x_min = min(x)
    x_max = max(x)
    y_ori = y*(x_max-x_min) + x_min
    return y_ori


y1 = scale_back(Propers[:, 0], ytest[:, 0])[:, np.newaxis]
y2 = scale_back(Propers[:, 1], ytest[:, 1])[:, np.newaxis]
y3 = scale_back(Propers[:, 2], ytest[:, 2])[:, np.newaxis]
'''
y4 = scale_back(Propers[:, 3], ytest[:, 3])[:, np.newaxis]
y5 = scale_back(Propers[:, 4], ytest[:, 4])[:, np.newaxis]
y6 = scale_back(Propers[:, 5], ytest[:, 5])[:, np.newaxis]
y7 = scale_back(Propers[:, 6], ytest[:, 6])[:, np.newaxis]
y8 = scale_back(Propers[:, 7], ytest[:, 7])[:, np.newaxis]
y9 = scale_back(Propers[:, 8], ytest[:, 8])[:, np.newaxis]
y10 = scale_back(Propers[:, 9], ytest[:, 9])[:, np.newaxis]
y11 = scale_back(Propers[:, 10], ytest[:, 10])[:, np.newaxis]
y12 = scale_back(Propers[:, 11], ytest[:, 11])[:, np.newaxis]
y13 = scale_back(Propers[:, 12], ytest[:, 12])[:, np.newaxis]
y14 = scale_back(Propers[:, 13], ytest[:, 13])[:, np.newaxis]
y15 = scale_back(Propers[:, 14], ytest[:, 14])[:, np.newaxis]
'''
yy = np.hstack((y1, y2, y3,)) #y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15))
pd.DataFrame(yy).to_csv('./result/y_test_true.csv', index=None, header=None)


y1 = scale_back(Propers[:, 0], ytrain[:, 0])[:, np.newaxis]
y2 = scale_back(Propers[:, 1], ytrain[:, 1])[:, np.newaxis]
y3 = scale_back(Propers[:, 2], ytrain[:, 2])[:, np.newaxis]
'''
y4 = scale_back(Propers[:, 3], ytrain[:, 3])[:, np.newaxis]
y5 = scale_back(Propers[:, 4], ytrain[:, 4])[:, np.newaxis]
y6 = scale_back(Propers[:, 5], ytrain[:, 5])[:, np.newaxis]
y7 = scale_back(Propers[:, 6], ytrain[:, 6])[:, np.newaxis]
y8 = scale_back(Propers[:, 7], ytrain[:, 7])[:, np.newaxis]
y9 = scale_back(Propers[:, 8], ytrain[:, 8])[:, np.newaxis]
y10 = scale_back(Propers[:, 9], ytrain[:, 9])[:, np.newaxis]
y11 = scale_back(Propers[:, 10], ytrain[:, 10])[:, np.newaxis]
y12 = scale_back(Propers[:, 11], ytrain[:, 11])[:, np.newaxis]
y13 = scale_back(Propers[:, 12], ytrain[:, 12])[:, np.newaxis]
y14 = scale_back(Propers[:, 13], ytrain[:, 13])[:, np.newaxis]
y15 = scale_back(Propers[:, 14], ytrain[:, 14])[:, np.newaxis]
'''
yy = np.hstack((y1, y2, y3,)) #y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15))
pd.DataFrame(yy).to_csv('./result/y_train_true.csv', index=None, header=None)


y1 = scale_back(Propers[:, 0], y_test_pred[:, 0])[:, np.newaxis]
y2 = scale_back(Propers[:, 1], y_test_pred[:, 1])[:, np.newaxis]
y3 = scale_back(Propers[:, 2], y_test_pred[:, 2])[:, np.newaxis]
'''
y4 = scale_back(Propers[:, 3], y_test_pred[:, 3])[:, np.newaxis]
y5 = scale_back(Propers[:, 4], y_test_pred[:, 4])[:, np.newaxis]
y6 = scale_back(Propers[:, 5], y_test_pred[:, 5])[:, np.newaxis]
y7 = scale_back(Propers[:, 6], y_test_pred[:, 6])[:, np.newaxis]
y8 = scale_back(Propers[:, 7], y_test_pred[:, 7])[:, np.newaxis]
y9 = scale_back(Propers[:, 8], y_test_pred[:, 8])[:, np.newaxis]
y10 = scale_back(Propers[:, 9], y_test_pred[:, 9])[:, np.newaxis]
y11 = scale_back(Propers[:, 10], y_test_pred[:, 10])[:, np.newaxis]
y12 = scale_back(Propers[:, 11], y_test_pred[:, 11])[:, np.newaxis]
y13 = scale_back(Propers[:, 12], y_test_pred[:, 12])[:, np.newaxis]
y14 = scale_back(Propers[:, 13], y_test_pred[:, 13])[:, np.newaxis]
y15 = scale_back(Propers[:, 14], y_test_pred[:, 14])[:, np.newaxis]
'''
yy = np.hstack((y1, y2, y3,)) #y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15))
pd.DataFrame(yy).to_csv('./result/y_test.csv', index=None, header=None)


y1 = scale_back(Propers[:, 0], y_train_pred[:, 0])[:, np.newaxis]
y2 = scale_back(Propers[:, 1], y_train_pred[:, 1])[:, np.newaxis]
y3 = scale_back(Propers[:, 2], y_train_pred[:, 2])[:, np.newaxis]
'''
y4 = scale_back(Propers[:, 3], y_train_pred[:, 3])[:, np.newaxis]
y5 = scale_back(Propers[:, 4], y_train_pred[:, 4])[:, np.newaxis]
y6 = scale_back(Propers[:, 5], y_train_pred[:, 5])[:, np.newaxis]
y7 = scale_back(Propers[:, 6], y_train_pred[:, 6])[:, np.newaxis]
y8 = scale_back(Propers[:, 7], y_train_pred[:, 7])[:, np.newaxis]
y9 = scale_back(Propers[:, 8], y_train_pred[:, 8])[:, np.newaxis]
y10 = scale_back(Propers[:, 9], y_train_pred[:, 9])[:, np.newaxis]
y11 = scale_back(Propers[:, 10], y_train_pred[:, 10])[:, np.newaxis]
y12 = scale_back(Propers[:, 11], y_train_pred[:, 11])[:, np.newaxis]
y13 = scale_back(Propers[:, 12], y_train_pred[:, 12])[:, np.newaxis]
y14 = scale_back(Propers[:, 13], y_train_pred[:, 13])[:, np.newaxis]
y15 = scale_back(Propers[:, 14], y_train_pred[:, 14])[:, np.newaxis]
'''
yy = np.hstack((y1, y2, y3,)) #y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15))
pd.DataFrame(yy).to_csv('./result/y_train.csv', index=None, header=None)

pd.DataFrame(compound_train).to_csv('./result/compound_train.csv', index=None, header=None)
pd.DataFrame(compound_test).to_csv('./result/compound_test.csv', index=None, header=None)