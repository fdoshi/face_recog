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
    new,cont,heir=cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    marker32=np.int32(marker)
    cv2.watershed(img,marker32)
    m=cv2.convertScaleAbs(marker32)
    ret,thresh=cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh_inv=cv2.bitwise_not(thresh)
    res=cv2.bitwise_and(img,img,mask=thresh)
    res2=cv2.bitwise_and(img,img,mask=thresh_inv)
    res3=cv2.addWeighted(res,1,res2,1,0)
    final=cv2.drawContours(res3,cont,-1,(0,0,255),1)
    # Display the resulting frame
    # cv2.imshow('bg',bg)
    # cv2.imshow('fg',fg)
    cv2.imshow('old',final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

   
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
