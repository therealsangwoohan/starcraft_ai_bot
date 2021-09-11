import tensorflow as tf
import keras.backend as backend
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import os
import random
import time


def getSession(gpuFraction=0.85):
    gpuOptions = tf.GPUOptions(per_process_gpu_memory_fraction=gpuFraction)
    return tf.Session(config=tf.ConfigProto(gpuOptions=gpuOptions))
backend.set_session(getSession())


model = Sequential()
model.add(Conv2D(32, (7, 7), padding='same',
                 input_shape=(176, 200, 1),
                 activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(14, activation='softmax'))

learningRate = 0.001
opt = keras.optimizers.adam(lr=learningRate)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/STAGE2-{}-{}".format(int(time.time()), learningRate))

trainDataDir = "trainData"

model = keras.models.load_model('BasicCNN-5000-epochs-0.001-LR-STAGE2')


def checkData(choices):
    totalData = 0

    lengths = []
    for choice in choices:
        print("Length of {} is: {}".format(choice, len(choices[choice])))
        totalData += len(choices[choice])
        lengths.append(len(choices[choice]))

    print("Total data length now is:", totalData)
    return lengths


hmEpochs = 5000

for i in range(hmEpochs):
    current = 0
    increment = 50
    notMaximum = True
    allFiles = os.listdir(trainDataDir)
    maximum = len(allFiles)
    random.shuffle(allFiles)

    while notMaximum:
        try:
            print("WORKING ON {}:{}, EPOCH:{}".format(current, current+increment, i))

            choices = {0: [],
                       1: [],
                       2: [],
                       3: [],
                       4: [],
                       5: [],
                       6: [],
                       7: [],
                       8: [],
                       9: [],
                       10: [],
                       11: [],
                       12: [],
                       13: [],
                       }

            for file in allFiles[current:current+increment]:
                try:
                    fullPath = os.path.join(trainDataDir, file)
                    data = np.load(fullPath)
                    data = list(data)
                    for d in data:
                        choice = np.argmax(d[0])
                        choices[choice].append([d[0], d[1]])
                except Exception as e:
                    print(str(e))

            lengths = checkData(choices)

            lowestData = min(lengths)

            for choice in choices:
                random.shuffle(choices[choice])
                choices[choice] = choices[choice][:lowestData]

            checkData(choices)

            trainData = []

            for choice in choices:
                for d in choices[choice]:
                    trainData.append(d)

            random.shuffle(trainData)
            print(len(trainData))

            testSize = 100
            batchSize = 128  # 128 best so far.

            xTrain = np.array([i[1] for i in trainData[:-testSize]]).reshape(-1, 176, 200, 1)
            yTrain = np.array([i[0] for i in trainData[:-testSize]])

            xTrain = np.array([i[1] for i in trainData[-testSize:]]).reshape(-1, 176, 200, 1)
            yTrain = np.array([i[0] for i in trainData[-testSize:]])

            model.fit(xTrain, yTrain,
                      batchSize=batchSize,
                      validation_data=(xTrain, yTrain),
                      shuffle=True,
                      epochs=1,
                      verbose=1, callbacks=[tensorboard])

            model.save("BasicCNN-5000-epochs-0.001-LR-STAGE2")
        except Exception as e:
            print(str(e))
        current += increment
        if current > maximum:
            notMaximum = False