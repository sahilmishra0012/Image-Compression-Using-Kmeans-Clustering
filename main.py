import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

input("Enter the file name with location")

img=mpimg.imread('/kaggle/input/image.jpg')

imgplot = plt.imshow(img)