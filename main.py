import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from functions import centroid_initialization,preprocess_image,optimize

#input("Enter the file name with location")

img=mpimg.imread('image.jpg')

img_new=preprocess_image(img)

k=3

centroids=centroid_initialization(img_new,k)

centroids_dictionary,centroids=optimize(img_new,centroids,3)

img_compressed=np.zeros(img_new.shape)

for key, value in centroids_dictionary.items():
    img_compressed[key]=centroids[value]
    
img_final=img_compressed.reshape((135, 165, 3))

imgplot = plt.imshow(img_final)

mpimg.msave('New_compressed.jpg')