import tensorflow as tf
import numpy
from tensorflow.python.keras._impl.keras.models import Sequential
from tensorflow.python.keras._impl.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras._impl.keras import optimizers
from matplotlib import pyplot
import pickle        

numpy.set_printoptions(threshold=numpy.nan)

for files in range (30,180,30):
    X = pickle.load(open("X"+str(files)+"color.pickle", "rb"))
    y = pickle.load(open('y'+str(files)+'color.pickle', 'rb'))

    X = X/255.0

    model = Sequential()
    
    model.add(Conv2D(128, (3, 3), input_shape = X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

##    model.add(Conv2D(128, (3, 3)))
##    model.add(Activation('relu'))
##    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
##    model.add(Dense(64))
##    model.add(Activation('relu'))

    model.add(Dense(64))#, activation = 'sigmoid'))
    model.add(Activation('sigmoid'))

    model.compile(loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'mae', 'mse'], optimizer='sgd')

    for l_r in range (1, 2, 1):
        l_r /= 8
        
        for mom in range (1, 2, 1):
            mom /= 8
            optimizers.SGD(lr = l_r, momentum = mom)

            history = model.fit(X, y, batch_size=300, epochs=1, validation_split=0.1)
            
            weights = [layer.get_weights() for layer in model.layers]

            with open('output.txt', 'a') as f, open('weights.txt', 'a') as f2:
                print(' ', file = f)
                print('For File: ' + str(files) + " color", file=f)
                print(' ', file = f2)
                print('For File: ' + str(files) + " color", file=f2)
                print(weights, file = f2)
                print('Epochs: 2', file = f)
                print('Learning Rate: ' +str(l_r), file = f)
                print('Momentum: ' +str(mom), file = f)
                print('Hidden Layers: 2', file = f)
                print('Mean Squared Error: '+str(history.history['mean_squared_error']), file=f)
                print('Mean Absolute Error: '+str(history.history['mean_absolute_error']), file = f)
                print('Accuracy: '+str(history.history['val_acc']), file = f)
                

##    pyplot.plot(history.history['mean_squared_error'])
##    pyplot.plot(history.history['mean_absolute_error'])
##    pyplot.plot(history.history['val_acc'])
##    pyplot.show()

                      


    
