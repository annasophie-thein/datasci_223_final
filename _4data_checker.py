"""
This script checks the dimensions of the .hdf5 files. 
"""
import h5py
import numpy as np

homedir = "/Users/anna-sophiethein/Dropbox/Medicine MAS/24WI DATASCI 223/github_223/datasci_223/final"

imageSize = 128  

datadir = homedir + '/segmented/train/' 
h5file = homedir + '/segmented_train'+str(imageSize)+'.hdf5'

# Open the HDF5 file for reading
with h5py.File(h5file, 'r') as f:
    # List all datasets in the file
    print("Datasets in the HDF5 file:")
    for dataset_name in f.keys():
        print(dataset_name)
        
    # Access and inspect a specific dataset (e.g., X_train)
    X_train = f['X_train']
    print("\nShape of X_train dataset:", X_train.shape)
    print("Example data from X_train:")
    print(X_train[0])  # Print an example data point from X_train

with h5py.File(h5file, 'r') as f:
    # Access and inspect the y_train dataset
    y_train = f['y_train']
    
    # Print information about the y_train dataset
    print("\nShape of y_train dataset:", y_train.shape)
    print("Example data from y_train:")
    print(y_train[0])  # Print an example label from y_train

    # Get unique values of y_test
    unique_labels_train = np.unique(y_train)
    print("Unique values of y_train:", unique_labels_train)



datadir = homedir + '/segmented/test/' 
h5file = homedir + '/segmented_test'+str(imageSize)+'.hdf5'

# Open the HDF5 file for reading
with h5py.File(h5file, 'r') as f:
    # List all datasets in the file
    print("Datasets in the HDF5 file:")
    for dataset_name in f.keys():
        print(dataset_name)
        
    # Access and inspect a specific dataset (e.g., X_test)
    X_test = f['X_test']
    print("\nShape of X_test dataset:", X_test.shape)
    print("Example data from X_test:")
    print(X_test[0])  # Print an example data point from X_test

# Open the HDF5 file for reading
with h5py.File(h5file, 'r') as f:
    # Access and inspect the y_test dataset
    y_test = f['y_test']
    
    # Print information about the y_test dataset
    print("\nShape of y_test dataset:", y_test.shape)
    print("Example data from y_test:")
    print(y_test[0])  # Print an example label from y_test

    # Get unique values of y_test
    unique_labels_test = np.unique(y_test)
    print("Unique values of y_test:", unique_labels_test)
