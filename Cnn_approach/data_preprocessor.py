
import cv2
import os
import numpy as np


IMAGE_SIZE =(154, 102)
# Load eye images and labels from a folder

X = [] # List of images
y = [] # List of labels

for folder in os.listdir("eye_images"): # Loop over folders in eye_images directory
    label = int(folder) # Assign a label based on folder name (0-9)
    for file in os.listdir("eye_images/" + folder): # Loop over files in each folder
        image = cv2.imread("eye_images/" + folder + "/" + file) 
        
        image = cv2.resize(image, (IMAGE_SIZE[0], IMAGE_SIZE[1])) # Resize image to fixed size
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert image to grayscale
        image = image / 255.0 # Normalize pixel values between 0 and 1
        
        X.append(image) 
        y.append(label) 

np.save('eye_data.npy', X)
np.save('eye_labels.npy', y)
