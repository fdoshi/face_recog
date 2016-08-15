import cv2
import numpy as np

layer_size = np.int32([62500, 32, 16, 4])
neural = cv2.ANN_MLP()
neural.create(layer_size)
neural.load('mlp.xml')
face_cas = cv2.CascadeClassifier("./haarcascade_frontalface_alt_tree.xml")
capture = cv2.VideoCapture(0)
storing_data = True

while storing_data:
    detect  = False
    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray)
    #gray = cv2.flip(frame, 1)
    encode = [int(cv2.IMWRITE_JPEG_QUALITY), 150]
    result, imgencode = cv2.imencode('.jpg', gray, encode)
    data = np.array(imgencode)
    #LOAD_IMAGE_GRAYSCALE converts the image iinto a 2-D matrix of grayscale.
    decimg = cv2.imdecode(data, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    #test = decimg

    for (x,y,w,h) in faces:
        detect = True
        cv2.rectangle(frame, (x,y+10),(x+w,y+h+20),(255,0,0))
        test = decimg[y+10:y+h+20,x:x+w]
        # r = 112 / test.shape[1]
        dim = (250 , 250)
        test = cv2.resize(test, dim, interpolation = cv2.INTER_AREA)
        unroll = test.reshape(1, 62500).astype(np.float32)
        cv2.imshow("hello2", test)

    if detect == True:
        try:
            ret, resp = neural.predict(unroll)
            print resp[0][resp.argmax(-1)]
            predict = resp.argmax(-1)
            #value = predict[0]


        except:
            print "Error"

        if predict[0] == 0:
            print "Prediction : fenil"
        elif predict[0] == 1:
            print "Prediction : rohit"
        elif predict[0] == 2:
            print "Predicted : mam"
        elif predict[0] == 3:
            print "Predicted : sir"
        else:
            print "Unknown"
    cv2.imshow("hello", frame)

    if(cv2.waitKey(30)==27&0xff):
        break

cv2.destroyAllWindows()

