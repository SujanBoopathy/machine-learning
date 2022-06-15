import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np

image=cv2.imread('nature.jpg')

plt.imshow(image)

image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

pixel_vals=image.reshape((-1,3))
pixel_vals=np.float32(pixel_vals)

criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10000,0.999)

k=3

retval,label,centers=cv2.kmeans(pixel_vals,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

centers=np.uint8(centers)

segmented_data=centers[label.flatten()]
segmented_img=segmented_data.reshape(image.shape)
plt.imshow(segmented_img)
plt.show()
