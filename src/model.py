from keras.models import Sequential
from keras.layers import Dense, Dropout , Activation, Embedding
from keras.layers import LSTM


def getModel2(inputShape, classes):
    model = Sequential()
    model.add(LSTM(classes, input_shape=inputShape))
    model.add(Dropout(0.1))
    model.add(Dense(classes, activation='softmax'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())




    return model



def getModel3(inputShape, classes):
    model = Sequential()
    model.add(Embedding(input_dim = 100, output_dim = 16, input_length = 1000))
    model.add(LSTM(classes,input_shape=inputShape ,activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(LSTM(classes, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation = 'sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model



def getModel(inputShape, classes):
    model = Sequential()
    model.add(LSTM(classes, init='uniform', inner_init='uniform',forget_bias_init='one', activation='tanh', inner_activation='sigmoid', input_shape=inputShape))
    model.add(Dropout(0.2)) 
    model.add(Dense(classes, activation='softmax'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model