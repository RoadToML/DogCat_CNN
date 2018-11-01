import tensorflow as tf
import numpy
import webbrowser 
from tensorflow.python.keras._impl.keras.models import Sequential
from tensorflow.python.keras._impl.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras._impl.keras import optimizers
import matplotlib.pyplot as plt
import matplotlib.colors as cm
import pickle        

numpy.set_printoptions(threshold = numpy.nan)

def plot_filters(layer, x, y):
    
    filters = layer.get_weights()
    fig = plt.figure()
    
    for j in range(len(filters)):
        temp_arr = filters[j].shape
        two_d_arr = temp_arr[:2]
        print(two_d_arr)
        ax = fig.add_subplot(y, x, 2)
        ax.matshow(two_d_arr[0][1], cmap = plt.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))

    plt.tight_layout()
    return plt
    


for files in range (20,40,20):
    X = pickle.load(open("X"+str(files)+"bw.pickle", "rb"))
    y = pickle.load(open('y'+str(files)+'bw.pickle', 'rb'))

    X = X/255.0

    model = Sequential()

    #  filter = 20, kernel = 3, strides = 1, input_shape = (X.shape[1:])
    model.add(Conv2D(32, 3, strides = 2, activation = 'relu', input_shape = (20, 20, 1)))
    model.add(MaxPooling2D(pool_size=(1,1))) #  strides = x

    #  filter = 20, kernel = 3, strides = 1
    model.add(Conv2D(32, 3, strides = 2, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(1,1))) #  strides = x

    model.add(Flatten())
    model.add(Dense(2, activation = 'relu'))

    model.compile(loss='sparse_categorical_crossentropy',
                  metrics=['accuracy', 'mae', 'mse'], optimizer='sgd')

    for l_r in range (1, 4, 1):
        l_r /= 10
        
        for mom in range (1, 4, 1):
            mom /= 10
            
            optimizers.SGD(lr = l_r, momentum = mom)

            history = model.fit(X, y, batch_size=5, epochs=5, validation_split=0.1)
            
            weights = [layer.get_weights() for layer in model.layers]

##            with open('output.txt', 'a') as f, open('weights.txt', 'a') as f2:
##                print(' ', file = f)
##                print('For File: ' + str(files) + " bw", file=f)
##                print(' ', file = f2)
##                print('For File: ' + str(files) + " bw", file=f2)
##                print(weights, file = f2)
##                print('=============================================================================', file = f2)
##                print('Epochs: 5', file = f)
##                print('Learning Rate: ' +str(l_r), file = f)
##                print('Momentum: ' +str(mom), file = f)
##                print('Hidden Layers: 2', file = f)
##                print('Mean Squared Error: '+str(history.history['mean_squared_error']), file=f)
##                print('Mean Absolute Error: '+str(history.history['mean_absolute_error']), file = f)
##                print('Accuracy: '+str(history.history['val_acc']), file = f)
##                print('Strides: 2', file = f)
##                print('Filter: 1', file = f)
##                print('Kernel: 3', file = f)
##                print('Activation: relu', file = f)
##                print('Pool Size: 1', file =f)
##
##webbrowser.open('output.txt')  

plot_filters(model.layers[0], 8, 4)

##    pyplot.plot(history.history['mean_squared_error'])
##    pyplot.plot(history.history['mean_absolute_error'])
##    pyplot.plot(history.history['val_acc'])
##    pyplot.show()

    
