import sys
import numpy as np
import cv2
from PIL import Image
import glob
image_list = []
count=1
for filename in glob.glob(sys.argv[1]+"*.jpg"):
    #im=Image.open(filename)
    #image_list.append(im)
    img = cv2.imread(filename)
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = sys.argv[3]
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
     # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    cv2.imwrite(sys.argv[2]+"image%d.jpg" % count, res2)
    count+=1
    #cv2.imshow('res2',res2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()