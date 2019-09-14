import numpy as np
from tqdm import tqdm

def preprocess_image(image):
    image=image/255
    img_new=np.reshape(image, (image.shape[0] * image.shape[1], image.shape[2]))
    return img_new

def centroid_initialization(image,k):
    m,n=image.shape
    centroids=np.zeros((k,n))
    for i in range(k): 
        centroids[i] = image[int(np.random.random(1)*1000)]
    return centroids

def distance(x1,y1,z1,x2,y2,z2): 
    dist = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2)+np.square(z1 - z2))
    return dist

def nearest_centroid(image,centroid):
    k=centroid.shape[0]
    centroids_dictionary={}
    for i in range(image.shape[0]):
        dlist=[]
        for j in range(k):
            dlist.append(distance(image[i][0],image[i][1],image[i][2],centroid[j][0],centroid[j][1],centroid[j][2]))
        min_dist=np.argmin(dlist)
        centroids_dictionary[i]=min_dist   
    return centroids_dictionary

def optimize(image,centroid,iterations):
    centroid_dictionary = nearest_centroid(image,centroid)
    for lo in tqdm(range(iterations)):
        print('Epoch'+str(lo))
        for key,value in centroid_dictionary.items():
            s=np.zeros((3,))
            count=0
            for i in range(image.shape[0]):
                if centroid_dictionary[i]==value:
                    s=s+image[i]
                    count=count+1
            s=s/count
            centroid[value]=s
        centroid_dictionary = nearest_centroid(image,centroid)
    return centroid_dictionary

