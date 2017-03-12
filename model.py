# Create the model.py
# 

import numpy as np
import csv
import cv2

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Lambda,Cropping2D
from keras.layers.core import  Dropout

from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


lines = []

# dirname = "/Users/vivek/Downloads/carnd/" First one 
# dirname = "/Users/vivek/Documents/carnd-data/" Second one
# dirname = "/Users/vivek/carnd/2-tracks/"
dirname = "/Users/vivek/carnd/stitching-different/"
image_dir = "IMG/"
csv_fname = "driving_log.csv"

with open(dirname + csv_fname) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = dirname+ image_dir + filename
    print(".",)
    image = cv2.imread(current_path)
    images.append(image)
    measurement=float(line[3])
    measurements.append(measurement)


X_train = np.array(images)


# plt.title("Image")
# print(X_train[0].shape)
# plt.imshow(X_train[0])
# plt.show()

# chopped = X_train[0][50:140,]
# plt.imshow(chopped)
# plt.show()
# # plt.imshow(X_train[2])
# exit()

y_train = np.array(measurements)
    
model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True,nb_epoch=3)

model.save('model.h5')

import gc; gc.collect()
