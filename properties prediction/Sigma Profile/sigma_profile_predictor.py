import numpy as np
from keras.layers import Dense, Input, LSTM, Embedding
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, TensorBoard
import subfunc_1 as subs
import pandas as pd
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

K.clear_session()

data_split = 0.8
R2_bound = 0.8
# select data
sigma_train = pd.read_csv('./DataPool/sigma.csv', index_col=None, header=None).values
compound_train0 = list(pd.read_csv('./DataPool/compound.csv', index_col=None, header=None)[0])
compound_test0 = pd.read_csv('./DataPool/select data/compound_test.csv', index_col=None, header=None)
smile_train = pd.read_csv('./DataPool/smiles.csv', index_col=None, header=None).values
smile_test = pd.read_csv('./DataPool/select data/smile_test.csv', index_col=None, header=None).values
'''
xtrain = smile_train[:, :50]
xtest = smile_test[:, :50]
ytrain = sigma_train[:, :51]
ytest = sigma_test[:, :51]
compound_train = list(compound_train0[0])
compound_test = list(compound_test0[0])
'''

xtrain, xtest = train_test_split(smile_train, test_size=0.2, random_state=33)
ytrain, ytest = train_test_split(sigma_train, test_size=0.2, random_state=33)
compound_train, compound_test = train_test_split(compound_train0, test_size=0.2, random_state=33)

# sigma calculator training

# choose embedding layer model
for i in range(1, 2):
    if i <= 1:
        output_dim = 500      # i * 10
    elif 1 < i <= 2:
        output_dim = 500
    else:
        output_dim = 800
    K.clear_session()
    print("i = ", i)
    model_name = './model_save/sigma_Dense_model_171_40k_test.h5'
    optimizer = Adam(0.001, 0.9, clipnorm=1)
    embedding = load_model('./model_save/embedding171K_MA & MO & T500_200.h5', custom_objects={'r2':subs.r2})
    x_embedding = embedding.predict(xtrain)
    x_embedding_test = embedding.predict(xtest)
    # Input
    input_shape = Input(batch_shape=(None, 50, output_dim))
    # Embedding
    LSTM1 = LSTM(units=500, return_sequences=False, stateful=False, activation='tanh', dropout=0.5,
                 recurrent_dropout=0.5)(input_shape)
    Dense1 = Dense(units=1000, activation='relu')(LSTM1)
    Dense1 = Dense(units=1000, activation='relu')(Dense1)

    calculator = Dense(units=51, activation='relu', name='sigma')(Dense1)

    Dense_model = Model(input=input_shape, output=calculator)
    Dense_model.summary()
    # Callbacks
    #visualization_callback = TensorBoard(write_graph=True, write_images=True)
    EarlyStopping_callback = EarlyStopping(patience=100, monitor='val_loss', mode='min')
    CSVLogger_callback = CSVLogger('./logs.csv')
    weight_save_callback = ModelCheckpoint(model_name,
                                           monitor='val_loss', verbose=0,
                                           save_best_only=True, mode='min', period=1)

    # model compile
    Dense_model.compile(optimizer=optimizer, loss='mse', metrics=[subs.r2])
    Dense_model.fit(x_embedding, ytrain,
                    nb_epoch=5000,
                    batch_size=100,
                    validation_data=(x_embedding_test[:64], ytest[:64]),
                    shuffle=True, callbacks=[CSVLogger_callback, weight_save_callback, EarlyStopping_callback,
                                             ])

    # save model
    Dense_model.save(model_name)

    # test
#model_name = './model_save/sigma_Dense_model_171_40k.h5'
model_name = './model_save/sigma_Dense_model_171_40k_4D.h5'
output_dim = 500
embedding = load_model('./model_save/embedding171K_MA & MO & T500_200.h5')
x_embedding = embedding.predict(xtrain)
x_embedding_test = embedding.predict(xtest)
Dense_model = load_model(model_name, custom_objects={'r2': subs.r2})
    # Dense_model = load_model('sigma_Dense_model.h5', custom_objects={'r2': subs.r2})
y_train_pred = Dense_model.predict(x_embedding)
y_test_pred = Dense_model.predict(x_embedding_test)
loss = Dense_model.evaluate(x=x_embedding_test, y=ytest, batch_size=2000)
print('\ntest loss:', loss)


    # picture output
SW_pic_output = 1
r2_c_train = subs.draw_sigma(ytrain, y_train_pred, compound_train, 'Train', SW_pic_output, '4D',
                                 bound=R2_bound)
r2_c_test = subs.draw_sigma(ytest, y_test_pred, compound_test, "Test", SW_pic_output, '4D', bound=R2_bound)
