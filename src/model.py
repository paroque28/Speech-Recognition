from keras.models import Sequential
from keras.layers import Dense, Dropout , Activation
from keras.layers import LSTM


def getModel(inputShaple, classes):
    model = Sequential()
    model.add(LSTM(classes, input_shape=inputShaple,return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(classes, activation='softmax'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())




    return model