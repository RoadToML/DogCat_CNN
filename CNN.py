import tensorflow as tf
from tensorflow.python.keras._impl.keras.models import Sequential
from tensorflow.python.keras._impl.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle

for file in range (50,100,50):
    X = pickle.load(open("X"+str(file)+".pickle", "rb"))
    y = pickle.load(open('y'+str(file)+'.pickle', 'rb'))

    X = X/255.0

    model = Sequential()
    model.add(Conv2D(128, (3, 3), input_shape = X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(X, y, batch_size=32, epochs=1, validation_split=0.1)
