import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    img_32=np.float32(img)
    meanshift=cv2.pyrMeanShiftFiltering(img,sp=8,sr=16,maxLevel=1,termcrit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,5,1))
    # Display the resulting frame
    cv2.imshow('frame',meanshift)
    cv2.imshow('old',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

   
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
