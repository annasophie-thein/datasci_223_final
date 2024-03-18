
import multiprocessing
import os
import cv2
import numpy as np
import time
t0 = time.time()

max_threads = 8

homedir = "/Users/anna-sophiethein/Dropbox/Medicine MAS/24WI DATASCI 223/github_223/datasci_223/final"

train = 1

if(train):
    train_dir = homedir + '/dataset/train/'
    store_dir = homedir + '/segmented/train/'
else:
    train_dir = homedir + '/dataset/test/'
    store_dir = homedir + '/segmented/test/'

def segment(path):
    #while True:
    #job = q.get()
    image_path = path[0]
    store_path = path[1]
    print("Processing image: %s"%image_path)
    # Read the image
    img = cv2.imread(image_path,0)
    if img is None:
        print(f"Failed to load image at {image_path}")
        return
    rows, cols = img.shape
    rows, cols = img.shape
    # Defining Ranges
    x = range(rows)
    y = range(cols)
    # Fill the White spots
    for xx in x:
        for yy in y:
            if (img[xx,yy]>245):
                img[xx,yy]=0
    # Apply Otsu's Binary Threshold    
    ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # Opening Operation ( Noise Reduction)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # Dilation Operation
    mask = cv2.dilate(opening,kernel,iterations=7)
    # initialize indice for locationg ROI
    indices = []
    # Loop through pixels to generate indices
    for xx in x:
        for yy in y:
            mm = mask[xx,yy]
            if (mm==0):
                indices.append(xx)
    # Find maximum and minimum region of interest
    minInd = min(indices)
    maxInd = max(indices)
    # Define spacing factor
    spacing = 60
    # Apply spacing factor to region of interest
    minInd = minInd - spacing
    # Round lower boundary
    if(minInd<0): minInd = 0
    # Upper boundary, Round
    maxInd = maxInd + spacing
    if(maxInd>rows): maxInd = rows
    # Read another clean instance of the image   
    backup = cv2.imread(image_path,0)
    # Crop the image
    cropped = backup[minInd:maxInd, :]
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    # Store the image
    cv2.imwrite(store_path,cropped)
    print("image segmented: %s"% store_path)
    #q.task_done()


paths = []
folders = os.listdir(train_dir)
for folder in folders:
    full_path = os.path.join(train_dir, folder)
    if os.path.isdir(full_path):
        images = os.listdir(full_path)
        for image in images:
            image_path  = os.path.join(full_path, image)
            store_path = os.path.join(store_dir, folder, image)
            paths.append([image_path,store_path])

if __name__ == '__main__':
    pool = multiprocessing.Pool(max_threads)
    pool.map(segment, paths)

t1 = time.time()
total = t1-t0
print("The script took a total of %d seconds to run"%total)