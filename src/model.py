from keras.models import Sequential
from keras.layers import Dense, Dropout , Activation
from keras.layers import LSTM



def getModel(inputShape, classes):
    model = Sequential()
    # model.add(LSTM(classes, init='uniform', inner_init='uniform',forget_bias_init='one', activation='tanh', inner_activation='sigmoid', input_shape=inputShape))
    model.add(LSTM(classes, input_shape=inputShape, bias_initializer="zeros", recurrent_dropout= 0.5))
    model.add(Dropout(0.3)) 
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model