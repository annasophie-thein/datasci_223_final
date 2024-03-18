"""
This script applies the segmenter function (segmenter_function.py) to images in the "test" folder
"""

import multiprocessing
import os
import cv2
import numpy as np
import time
from segmenter_function import segment

t0 = time.time()

max_threads = 8

homedir = "/Users/anna-sophiethein/Dropbox/Medicine MAS/24WI DATASCI 223/github_223/datasci_223/final"

train_dir = homedir + '/dataset/test/'
store_dir = homedir + '/segmented/test/'

paths = []

folders = os.listdir(train_dir)
for folder in folders:
    full_path = os.path.join(train_dir, folder)
    if os.path.isdir(full_path):
        images = os.listdir(full_path)
        for image in images:
            image_path  = os.path.join(full_path, image)
            store_path = os.path.join(store_dir, folder, image)
            paths.append((image_path,store_path))  # Use tuple instead of list

if __name__ == '__main__':
    with multiprocessing.Pool(max_threads) as pool:
        pool.starmap(segment, paths)  # Use starmap instead of map

t1 = time.time()
total = t1-t0
print("The script took a total of %d seconds to run"%total)
