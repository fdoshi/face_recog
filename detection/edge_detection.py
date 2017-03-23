import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ##erode the image
    fg=cv2.erode(thresh,None,iterations=1)
    ##dilate the image
    bgt=cv2.dilate(thresh,None,iterations=1)
    ret,bg=cv2.threshold(bgt,1,128,1)
    marker=cv2.add(fg,bg)
    canny=cv2.Canny(marker,110,150)
    # Display the resulting frame
    # cv2.imshow('bg',bg)
    # cv2.imshow('fg',fg)
    cv2.imshow('old',canny)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

   
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
