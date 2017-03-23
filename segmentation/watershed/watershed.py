import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import label

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    mask=np.zeros(frame.shape,np.uint8)
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    unproc_gray=gray
    ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    kernel=np.ones((3,3),np.uint8)
    opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
    # #background
    sure_bg=cv2.dilate(opening,kernel,iterations=3)
    dist_transform=cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret,sure_fg=cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    ##FINDING UNKNOWN
    sure_fg=np.uint8(sure_fg)
    unknown=cv2.subtract(sure_bg,sure_fg)
    ##creating markers
    ret,markers=cv2.connectedComponents(sure_fg)
    markers+=1
    markers[unknown==255]=0
    markers = cv2.watershed(frame,markers)
    gray[markers == -1] = 0
    frame[markers==-1]=[0,0,255]
    mask[markers==-1]=[0,0,255]





    # Display the resulting frame
    cv2.imshow('frame',unproc_gray)
    cv2.imshow('old',mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
# img=cv2.imread('coins.png')
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# print("done")

# ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# cv2.imshow('image',thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #removing the noise
# kernel=np.ones((3,3),np.uint8)
# opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

# #background
# sure_bg=cv2.dilate(opening,kernel,iterations=3)

# cv2.imshow('image',sure_bg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #foreground
# dist_transform=cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret,sure_fg=cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# ##FINDING UNKNOWN
# sure_fg=np.uint8(sure_fg)
# unknown=cv2.subtract(sure_bg,sure_fg)


# ##creating markers

# ret,markers=cv2.connectedComponents(sure_fg)
# markers+=1
# markers[unknown==255]=0

# markers = cv2.watershed(img,markers)
# img[markers == -1] = [255,0,0]

# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

