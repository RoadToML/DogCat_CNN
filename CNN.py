import tensorflow as tf
from tensorflow.python.keras._impl.keras.models import Sequential
from tensorflow.python.keras._impl.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from matplotlib import pyplot
import pickle

for files in range (50,100,50):
    X = pickle.load(open("X"+str(files)+".pickle", "rb"))
    y = pickle.load(open('y'+str(files)+'.pickle', 'rb'))

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
                  metrics=['accuracy', 'mae', 'mse'])

    for epoch in range(1, 5, 1):
        history = model.fit(X, y, batch_size=250, epochs=epoch, validation_split=0.1)
        with open('output.txt', 'a') as f:
            print(' ', file = f)
            print('For File: ' + str(files), file=f)
            print('Epochs: ' + str(epoch), file = f)
            print('Hidden Layers: 3', file = f)
            print('Mean Squared Error: '+str(history.history['mean_squared_error']), file=f)
            print('Mean Absolute Error: '+str(history.history['mean_absolute_error']), file = f)
            print('Accuracy: '+str(history.history['val_acc']), file = f)

##    pyplot.plot(history.history['mean_squared_error'])
##    pyplot.plot(history.history['mean_absolute_error'])
##    pyplot.plot(history.history['val_acc'])
##    pyplot.show()

    
    print('****************************************************************' \
          '****************************************************************' \
          '****************************************************************' \
          '****************************************************************' \
          '****************************************************************' \
          '****************************************************************' \
          '****************************************************************' \
          '****************************************************************' \
          '****************************************************************' \
          '****************************************************************' \
          '****************************************************************' \
          '****************************************************************' \
          '****************************************************************' \
          '****************************************************************' \
          '****************************************************************')
          


    
