"""
This script trains the model. 
"""
import h5py
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import math

# Set seed for reproducibility
seed = 7
np.random.seed(seed)

# Define parameters
num_classes = None  # Placeholder for the actual number of classes
imageSize = 128
epochs = 15
batch_size = 100

# Specify directory and models
model_name = 'segmented128'
homedir = "/Users/anna-sophiethein/Dropbox/Medicine MAS/24WI DATASCI 223/github_223/datasci_223/final"
model_dir = os.path.join(homedir, 'model', model_name)
weightFile_best = os.path.join(model_dir, 'best.keras')
h5file_train = os.path.join(homedir, 'segmented_train128.hdf5')
h5file_test = os.path.join(homedir, 'segmented_test128.hdf5')

# Function to preprocess labels and generate data batches
def generator(h5file, batch_size, num_classes):
    with h5py.File(h5file, 'r') as f:
        X_train = f['X_train']
        y_train = f['y_train']
        total_data = X_train.shape[0]
        steps_per_epoch = math.floor(total_data / batch_size)
        
        while True:
            start = 0
            for step in range(0, steps_per_epoch):
                end = start + batch_size
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]

                # Preprocess y_batch to ensure it contains values between 0 and num_classes - 1
                y_batch = np.clip(y_batch, 0, num_classes - 1)

                X_batch = X_batch / 255
                X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[1], X_batch.shape[2], 1)
                y_batch = to_categorical(y_batch, num_classes=num_classes)
                start = end
                yield X_batch, y_batch


# Load the test dataset
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
                
# Function to compute steps per epoch
def stepsCount(h5file, batch_size):
    with h5py.File(h5file, 'r') as f:
        X_train = f['X_train']
        total_data = X_train.shape[0]
        steps_per_epoch = math.floor(total_data / batch_size)
        return steps_per_epoch

# Compute the actual number of classes
with h5py.File(h5file_train, 'r') as f:
    y_train = f['y_train'][:]
    num_classes = len(np.unique(y_train))

# Compute steps per epoch
spe = stepsCount(h5file_train, batch_size)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(imageSize, imageSize, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Initialize callbacks
checkpoint = ModelCheckpoint(weightFile_best, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, verbose=1, mode='max')

# Train the model
history = model.fit(generator(h5file_train, batch_size, num_classes),
                    steps_per_epoch=spe,
                    epochs=epochs,
                    validation_data=(X_test, y_test),
                    callbacks=[checkpoint, early_stopping])

# Save the final model
model.save(os.path.join(model_dir, 'final_model.h5'))
