%reset -f

# <codecell>

import csv

lines = []
with open('training-data\\driving_log_track2.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

print("Finished reading CSV file.")

# <codecell>

import cv2
import progressbar

strA_corr = 0.2
images_aug , measurements_aug = [], []
bar = progressbar.ProgressBar(max_value=len(lines))
bar.start()
for i in range(len(lines)):
    line = lines[i]
    imgC = cv2.imread(line[0])
    imgL = cv2.imread(line[1])
    imgR = cv2.imread(line[2])
    strA = float(line[3])
    if abs(strA) < 0.01:
        continue
    images_aug.extend([
                       imgC, cv2.flip(imgC,1)
                       #imgL, cv2.flip(imgL,1),
                       #imgR, cv2.flip(imgR,1)
                       ])
    measurements_aug.extend([
                             strA,-1.0*strA
                             #strA+strA_corr,-1.0*(strA+strA_corr),
                             #strA-strA_corr,-1.0*(strA-strA_corr)
                             ])
    bar.update(i)
bar.finish()

# <codecell>

# clear unneeded variables
line = None ; lines = None ; i = None
imgC = None ; imgL = None ; imgR = None
strA = None ; strA_corr = None

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

images_aug, measurements_aug = shuffle(np.array(images_aug),
                                       np.array(measurements_aug))

X_train, X_valid, y_train, y_valid = train_test_split(images_aug,
                                                      measurements_aug,
                                                      test_size=0.2)

print("Training features shape  : {}".format(X_train.shape))
print("Training labels shape    : {}".format(y_train.shape))
print("Validation features shape: {}".format(X_valid.shape))
print("Validation labels shape  : {}".format(y_valid.shape))

# clear the augmented image arrays
images_aug = None ; measurements_aug = None

# <codecell>

def generator(X_data, y_data, batch_size=32):
    num_samples = len(X_data)
    while 1: # Loop forever so the generator never terminates
        shuffle(X_data, y_data)
        for offset in range(0, num_samples, batch_size):
            batch_x = X_data[offset:offset+batch_size]
            batch_y = y_data[offset:offset+batch_size]
        yield batch_x, batch_y

train_generator = generator(X_train, y_train, batch_size=32)
validation_generator = generator(X_valid, y_valid, batch_size=32)

# <codecell>

from os import path
from pickle import dump, load

# WARNING: The following will produce a rather large dump file due to JPG
#          decompression and raw format
#http://stackoverflow.com/questions/29704139/pickle-in-python3-doesnt-work-for-large-data-saving

dumpFlag = False

if dumpFlag:
    
    # Check for and dump training data
    if not path.exists("training-data.p"):
        print("Saving training data to pickle file ...")
        dump({"features": X_train, "labels": y_train},
             open("training-data.p", "wb"),
             protocol=4)
    else:
        print("Found training pickled file. Loading data ...")
        pdata = load(open("training-data.p", mode='rb'))
        X_train = pdata['features']
        y_train = pdata['labels']
    print("Training dataset size: {}".format(X_train.shape[0]))
    
    # Check for and dump Validation data
    if not path.exists("validation-data.p"):
        print("Saving validation data to pickle file ...")
        dump({"features": X_valid, "labels": y_valid},
             open("validation-data.p", "wb"),
             protocol=4)
    else:
        print("Found validation pickled file. Loading data ...")
        pdata = load(open("validation-data.p", mode='rb'))
        X_valid = pdata['features']
        y_valid = pdata['labels']
    print("Validation dataset size: {}".format(X_valid.shape[0]))

# <codecell>

from keras import backend as kbe
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

kbe.set_image_dim_ordering('tf')

model = Sequential()
model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(MaxPooling2D())
#model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# <codecell>

import matplotlib.pyplot as plt

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_data=(X_valid,y_valid),
          shuffle=True, nb_epoch=5, verbose=1)

#history_object = model.fit_generator(train_generator,
#                                     samples_per_epoch=len(X_train),
#                                     validation_data=validation_generator,
#                                     nb_val_samples=len(X_valid),
#                                     nb_epoch=5,
#                                     verbose=1)
### print the keys contained in the history object
#print(history_object.history.keys())
### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()

# <codecell>

model.save('model.h5')

#model_json = model.to_json()
#with open('model.json', 'w') as json_file:
#    json_file.write(model_json)
#model.save_weights("model_weights.h5")

print("Saved model to disk")

# <codecell>

#https://github.com/fchollet/keras/issues/3964
#https://github.com/fchollet/keras/issues/4044
#https://github.com/fchollet/keras/issues/4143
#https://github.com/fchollet/keras/issues/4802
#https://github.com/fchollet/keras/issues/5442 (*)
#http://machinelearningmastery.com/save-load-keras-deep-learning-models/

# Run model network creation step (without compile & fit) before loading model
model = load_model('model.h5')

#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#model_1 = model_from_json(loaded_model_json)
#model_1.load_weights("model_weights.h5")

print("Loaded model from disk")

# <codecell>

runfile('drive.py', args='model_track2.h5 video_track2')

# <codecell>

runfile('video.py', args='video_track2 --fps 48')
