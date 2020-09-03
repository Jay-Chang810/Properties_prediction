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

data_split = 0.8
R2_bound = 0.8
Proper = 'Pc'

Propers = pd.read_csv('./DataPool/Data3/Property.csv', index_col=None, header=None).values
compound= list(pd.read_csv('./DataPool/Data3/compound.csv', index_col=None, header=None)[0])
smiles = pd.read_csv('./DataPool/Data3/smiles.csv', index_col=None, header=None).values

scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
scaler.fit((Propers))
Propers1 = scaler.transform((Propers))

#propers0 = np.hstack((Propers[:, 0][:, np.newaxis], Propers[:, 3][:, np.newaxis]))
xtrain, xtest = train_test_split(smiles, test_size=0.2, random_state=10)
ytrain, ytest = train_test_split(Propers1, test_size=0.2, random_state=10)
compound_train, compound_test = train_test_split(compound, test_size=0.2, random_state=10)

# sigma calculator training

# choose embedding layer model
for i in range(1, 2):
    if i <= 1:
        output_dim = 500       # i * 10
    elif 1 < i <= 2:
        output_dim = 500
    else:
        output_dim = 800
    K.clear_session()
    print("i = ", i)
    load_model_name = './model_save/embedding171K_MA & MO & T500_200.h5'
    model_name = './model_save/' + Proper + '_' + str(output_dim) + '_171.h5'

    embedding = load_model(load_model_name, custom_objects={'r2':subs.r2})
    x_embedding = embedding.predict(xtrain)
    x_embedding_test = embedding.predict(xtest)
    # Input
    input_shape = Input(batch_shape=(None, 50, output_dim))
    # Embedding
    LSTM1 = LSTM(units=200, return_sequences=False, stateful=False, activation='tanh', dropout=0.5,
                 recurrent_dropout=0.5)(input_shape)
    Dense1 = Dense(units=200, activation='relu')(LSTM1)
    calculator = Dense(units=4, activation='relu', name='sigma')(Dense1)

    Dense_model = Model(input=input_shape, output=calculator)
    Dense_model.summary()
    # Callbacks
    #visualization_callback = TensorBoard(write_graph=True, write_images=True)
    EarlyStopping_callback = EarlyStopping(patience=150, monitor='val_loss', mode='min')
    CSVLogger_callback = CSVLogger('./logs.csv')
    weight_save_callback = ModelCheckpoint(model_name,
                                           monitor='val_loss', verbose=0,
                                           save_best_only=True, mode='min', period=1)

    # model compile
    Dense_model.compile(optimizer='Adam', loss='mae')
    Dense_model.fit(x_embedding, ytrain,
                    nb_epoch=5000,
                    batch_size=100,
                    validation_data=(x_embedding_test[:32], ytest[:32]),
                    shuffle=True, callbacks=[CSVLogger_callback, weight_save_callback, EarlyStopping_callback,
                                             ])

    # save model
    Dense_model.save(model_name)

output_dim = 500

# test
embedding = load_model(load_model_name)
x_embedding = embedding.predict(xtrain)
x_embedding_test = embedding.predict(xtest)
Dense_model = load_model(model_name, custom_objects={'r2': subs.r2})
    # Dense_model = load_model('sigma_Dense_model.h5', custom_objects={'r2': subs.r2})
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
y4 = scale_back(Propers[:, 3], ytest[:, 3])[:, np.newaxis]
yy = np.hstack((y1, y2, y3, y4))
pd.DataFrame(yy).to_csv('./result/y_test_true.csv', index=None, header=None)


y1 = scale_back(Propers[:, 0], ytrain[:, 0])[:, np.newaxis]
y2 = scale_back(Propers[:, 1], ytrain[:, 1])[:, np.newaxis]
y3 = scale_back(Propers[:, 2], ytrain[:, 2])[:, np.newaxis]
y4 = scale_back(Propers[:, 3], ytrain[:, 3])[:, np.newaxis]
yy = np.hstack((y1, y2, y3, y4))
pd.DataFrame(yy).to_csv('./result/y_train_true.csv', index=None, header=None)


y1 = scale_back(Propers[:, 0], y_test_pred[:, 0])[:, np.newaxis]
y2 = scale_back(Propers[:, 1], y_test_pred[:, 1])[:, np.newaxis]
y3 = scale_back(Propers[:, 2], y_test_pred[:, 2])[:, np.newaxis]
y4 = scale_back(Propers[:, 3], y_test_pred[:, 3])[:, np.newaxis]
yy = np.hstack((y1, y2, y3, y4))
pd.DataFrame(yy).to_csv('./result/y_test.csv', index=None, header=None)


y1 = scale_back(Propers[:, 0], y_train_pred[:, 0])[:, np.newaxis]
y2 = scale_back(Propers[:, 1], y_train_pred[:, 1])[:, np.newaxis]
y3 = scale_back(Propers[:, 2], y_train_pred[:, 2])[:, np.newaxis]
y4 = scale_back(Propers[:, 3], y_train_pred[:, 3])[:, np.newaxis]
yy = np.hstack((y1, y2, y3, y4))
pd.DataFrame(yy).to_csv('./result/y_train.csv', index=None, header=None)

pd.DataFrame(compound_train).to_csv('./result/compound_train.csv', index=None, header=None)
pd.DataFrame(compound_test).to_csv('./result/compound_test.csv', index=None, header=None)