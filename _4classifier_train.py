
#%%

import h5py
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Convolution2D
import math
#%%

#from keras import backend as K
#K.set_image_data_format('channels_last')

homedir = "/Users/anna-sophiethein/Dropbox/Medicine MAS/24WI DATASCI 223/github_223/datasci_223/final"

seed = 7
np.random.seed(seed)
num_classes = 4
imageSize = 128
epochs = 15
batch_size = 100

model_name = 'segmented128'

model_dir = homedir + '/model/' + model_name
weightFile_best = model_dir + '/best.hdf5'
h5file_train = homedir + '/segmented_train128.hdf5'
h5file_test = homedir + '/segmented_test128.hdf5'

#%%
def generator(h5file, batch_size):
    with h5py.File(h5file,'r') as f:
        X_train = f['X_train']
        y_train = f['y_train']
        total_data = X_train.shape[0]
        steps_per_epoch = math.floor(total_data/batch_size)
        
        while True:
            start = 0
            for step in range(0,steps_per_epoch):
                end = start + batch_size
                X_batch =  X_train[start:end]
                y_batch =  y_train[start:end]
                X_batch = X_batch/255
                X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], X_batch.shape[2], 1)
                y_batch = to_categorical(y_batch,num_classes = 4)
                start = end
                yield X_batch,y_batch


#%%

with h5py.File(h5file_test,'r') as f:
    X_test = f['X_test'][:]
    y_test = f['y_test'][:]
    X_test = X_test/255
    print(X_test.shape)  # print the shape of X_test
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    y_test = to_categorical(y_test)

with h5py.File(h5file_test,'r') as f:
    X_test = f['X_test'][:]
    y_test = f['y_test'][:]
    X_test = X_test/255
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    y_test = to_categorical(y_test)
                
#%%
    
def stepsCount(h5file, batch_size):
    with h5py.File(h5file,'r') as f:
        X_train = f['X_train']
        y_train = f['y_train']
        total_data = X_train.shape[0]
        steps_per_epoch = math.floor(total_data/batch_size)
        return steps_per_epoch

spe = stepsCount(h5file_train,batch_size)
                
#%%

model = Sequential()

model = Sequential()
model.add(Convolution2D(32, 3, 3 , input_shape=(imageSize,imageSize,1),activation= 'relu' ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation= 'relu' ))
model.add(Dense(num_classes, activation= 'softmax' ))

#%%

model.compile(loss= 'categorical_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])

# Update the weightFile and weightFile_best to end with .keras
weightFile = "/weights-improvement-{epoch:02d}-{val_acc:.2f}.keras"
weightFile_best = model_dir + '/best.keras'

# Create the ModelCheckpoint instances with the updated filepaths
#check3 = EarlyStopping(monitor = 'val_categorical_accuracy',min_delta = 0.00001 , patience = 2 , verbose = 0, mode = 'auto')
check2 = ModelCheckpoint((model_dir + weightFile), monitor = 'val_categorical_accuracy', verbose = 1)
check1  = ModelCheckpoint(weightFile_best, monitor = 'val_categorical_accuracy', verbose = 1)

# Create the list of callbacks
checkpoints = [check1,check2]

# Keep datagen in the model.fit call
datagen = generator(h5file_train, batch_size)
history = model.fit(datagen, steps_per_epoch=spe, epochs=epochs,validation_data = (X_test,y_test), verbose=1, callbacks=checkpoints)
#%%

# summarize history for accuracy
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(4,3))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
rootdir =homedir+ '/figures/acc.png'
plt.savefig(rootdir,figsize=(4,3),dpi=500,bbox_inches='tight',labelsize=12)
plt.show()

#%%

# summarize history for loss
fig = plt.figure(figsize=(4,3))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
rootdir =homedir+ '/figures/loss.png'
plt.savefig(rootdir,figsize=(4,3),dpi=500,bbox_inches='tight',labelsize=12)
plt.show()

#%%

y_pred = model.predict(X_test)

#%%
# confusion matrix plot
from numpy import argmax
#y_train=y_train.argmax(1)
y_test=argmax(y_test,axis = 1)
y_pred=argmax(y_pred,axis = 1)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

#%%

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#%%

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

#%%

# Plot non-normalized confusion matrix
plt.figure()
class_names=['NORMAL','CNV','DME','DRUSEN']
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
rootdir =homedir+ '/figures/confusion_matrix_without_normalization.png'
plt.savefig(rootdir,figsize=(4,3),dpi=500,bbox_inches='tight',labelsize=12)

#%%

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')
plt.legend(['train', 'test'], loc='lower right')
rootdir =homedir+ '/figures/confusion_matrix_with_normalization.png'
plt.savefig(rootdir,figsize=(4,3),dpi=500,bbox_inches='tight',labelsize=12)
plt.show()



