# Partly written by author: Siddhartha Gairola
# Substantially adaptee from References 1, 2 in Readme.txt file.

from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#Ignore Warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion() #interative
print("Import Successful ClassificationImageLoader")

def default_image_loader(image_file):                                            
    img = cv2.imread(image_file)
    
    if img is None:
        return np.zeros((224, 224, 3))
    # GRAYSCALE 
    if len(img.shape) == 2:                                                 
        img_new = np.ndarray( (img.shape[0], img.shape[1], 3), dtype = 'float32')
        img_new[:,:,0] = img                                                    
        img_new[:,:,1] = img                                                    
        img_new[:,:,2] = img                                                    
        img = img_new                                                           
    
    img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2RGB)                    
    img = img.astype('float32')                                             
    img_resized = cv2.resize(img, (224, 224))

    return (img_resized/255.0).astype('float32')

class ClassificationImageLoader(Dataset):

    def __init__(self, base_path, filenames_filename, labels_filename, transform=None, loader = default_image_loader):

        # filenames_filename => A text file with each line containing a path to an image, e.g., images/class1/sample.jpg
        # labels_filename => A text file with each line containing 1 integer, label index of the image.

        self.base_path = base_path
        self.filenamelist = []

        for line in open(filenames_filename):
            self.filenamelist.append(line.rstrip('\n'))

        labels = []

        for line in open(labels_filename):
            labels.append(int(line.strip())) # label

        self.labels = labels
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        #print((os.path.join(self.base_path,self.filenamelist[int(index)])), self.labels[index], index)
        img1 = self.loader(os.path.join(self.base_path,self.filenamelist[index]))

        if self.transform:
            img1 = self.transform(img1)

        return img1, self.labels[index]

    def __len__(self):
        return len(self.labels)
