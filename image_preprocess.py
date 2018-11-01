import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "C:/Users/black/Desktop/Uni Work/2018/Sem 2/NIC/Assignment 2/crossAndLines/Photoshop"
CATEGORIES = ["crosses", "lines"]

IMG_SIZE = 20

##for category in CATEGORIES:
##    path = os.path.join(DATADIR, category)
##    for img in os.listdir(path):
##        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
##        plt.imshow(img_array, cmap="gray")
##        plt.show()
##        break
##    break

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) #each folder is 1 animal
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE) # remove for colour
                #print img_array
                #for create_training_data.IMG_SIZE in range(50, 300, 50):
                #add IMG_ARRAY x IMg_ARRAY below and save the pickle file as X.IMG_ARRAY.pickle
                #then in the CNN file do the for loop for opening each file and making the net.
                new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

print(len(training_data))

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X = [] #feature set
y = [] #labels  -- both of these are similar to train_X, test_x etc

for features, label in training_data:
    X.append(features) #cant pass a list to the neural network, X has to be a numpy array
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) #-1 is any number, imgage size, +1 = grayscale


#save your data so you dont have to rebuild it everytime.
pickle_out = open("X"+str(IMG_SIZE)+"bw.pickle", 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y"+str(IMG_SIZE)+"bw.pickle", 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

#load data back 
pickle_in = open('X'+str(IMG_SIZE)+'bw.pickle', 'rb')
X = pickle.load(pickle_in)

X[1]
