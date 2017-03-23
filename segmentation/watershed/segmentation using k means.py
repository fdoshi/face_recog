import cv2
import numpy as np

import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import label

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    img_32=np.float32(img)
    conditions=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1)
    k=50
    ret,label,center=cv2.kmeans(img_32,k,None,conditions,50,cv2.KMEANS_RANDOM_CENTERS)
    center=np.uint8(center)
    res=center[label.flatten()]
    res2=res.reshape(img.shape)
    # Display the resulting frame
    cv2.imshow('frame',res2)
    cv2.imshow('old',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

   
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

