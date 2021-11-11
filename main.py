############################## Web Cam Code################################
# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
# cap.set(3,300)
# cap.set(4,300)
# cap.set(10,50)
# kernel=np.ones((5,5),np.uint8)
# while True:
#     success, img=cap.read()
#     cv2.line(img, (0, 0), (300, 300), (0, 255, 0), 3)
#     imgcanny=cv2.Canny(img,100,100)
#     imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     imgBlur = cv2.GaussianBlur(imgGray, (47, 47), 0)
#     imgDialation = cv2.dilate(imgcanny, kernel, iterations=1)
#     imgEroded = cv2.erode(imgDialation, kernel, iterations=1)
#
#     cv2.imshow("Orgnal video",img)
#     cv2.imshow("Video Edige",imgcanny)
#     cv2.imshow("Gray Video",imgGray)
#     cv2.imshow("Blur video",imgBlur)
#     cv2.imshow("Dialation Image",imgDialation)
#     cv2.imshow("Eroded Image", imgEroded)
#     cv2.imshow("Eroded Image", imgEroded)
#
#     print("Its Running")
#     if cv2.waitKey(1) & 0xFF ==ord('q'):
#         break
################################## web cam edge detecting ###############################
# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
#
# def getContours(imgcanny):
#      contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#      for cnt in contours:
#          area =cv2.contourArea(cnt)
#          print(area)
#          # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 2)
#
# while True:
#     success, img = cap.read()
#     # imgContour = img.copy()
#     imgcanny = cv2.Canny(img, 100, 100)
#     cv2.imshow("Video Edige", imgcanny)
# #    getContours(imgcanny)
#
#     if cv2.waitKey(1) & 0xFF ==ord('q'):
#         break
###############################Color image###################################
# import cv2
# import numpy as np
# img=cv2.imread("Resorce/pic.jpg")
# kernel=np.ones((5,5),np.uint8)
# imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgBlur=cv2.GaussianBlur(imgGray,(47,47),0)
# imgCanny=cv2.Canny(img,150,200)                          #Image eidg detection
# imgDialation = cv2.dilate(imgCanny,kernel,iterations=1)
# imgEroded = cv2.erode(imgDialation,kernel,iterations=1)
#
# cv2.imshow("Gray iamge",imgGray)
# cv2.imshow("Blure image",imgBlur)
# cv2.imshow("Edige Detecting",imgCanny)
# cv2.imshow("Image Delition",imgDialation)
# cv2.imshow("Eroded Image",imgEroded)
# cv2.waitKey(0)

############################### Resizeing & Croping ##########################
# import cv2
# import numpy as np
# img=cv2.imread("Resorce/pic.jpg")
#
# imgResize=cv2.resize(img,(462,623))
# cv2.imshow("image",img)
# cv2.imshow("Resize image",imgResize)
# print(img.shape)
# print(imgResize.shape)
# imgCropped =img[0:200,200:500]
# cv2.imshow("Cropped",imgCropped)
# cv2.waitKey(0)

############################## Shape and Texts #################################
# import cv2
# import numpy as np
# img=np.zeros((512,512,3),np.uint8)
# # print(img)
# # img[:]=255,0,0
# cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)
# cv2.rectangle(img,(0,0),(250,350),(0,0,255),2)
# cv2.circle(img,(400,50),30,(255,255,0),5)
# cv2.putText(img,"OpenCV",(300,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1)
# cv2.imshow("imgg",img)
# cv2.waitKey(0)

############################ Warp Presepective ################################
           # Error #
# import cv2
# import numpy as np
# img=cv2.imread("Resorce/pic.jpg")
# width,height=250,350
# pts1=np.float32([[111,219],[287,188][154,482],[352,440]])
# pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
# matrix=cv2.getPerspectiveTransform(pts1,pts2)
# imgOutput=cv2.warpPerspective(img,matrix,(width,height))
# cv2.imshow("warp Image",imgOutput)
# cv2.imshow("Image",img)
# cv2.waitKey(0)
############################ joining Images ####################################
# import cv2
# import numpy as np
# img=cv2.imread("Resorce/pic1.jpg")
# imgResize=cv2.resize(img,(462,623))
# imghor = np.hstack((imgResize,imgResize))
# imgver = np.vstack((imgResize,imgResize))
# #cv2.imshow("Horizental",imghor)
# cv2.imshow("Vertical",imgver)
# print(imgResize.shape)
# cv2.waitKey(0)

########################### Color Detection #####################################
# import cv2
# import numpy as np
# def empty(a):
#     pass
#
#
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars",640,340)
# cv2.createTrackbar("Hue Min","TrackBars",0,179,empty)
# cv2.createTrackbar("Hue Max","TrackBars",19,179,empty)
# cv2.createTrackbar("Sat Min","TrackBars",110,255,empty)
# cv2.createTrackbar("Sat Max","TrackBars",240,255,empty)
# cv2.createTrackbar("val Min","TrackBars",153,255,empty)
# cv2.createTrackbar("val Max","TrackBars",255,255,empty)
#
# while True:
#     img = cv2.imread("Resorce/pic1.jpg")
#     imgResize = cv2.resize(img, (462, 623))
#     h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
#     h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
#     s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
#     s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
#     v_min = cv2.getTrackbarPos("val Min", "TrackBars")
#     v_max = cv2.getTrackbarPos("val Max", "TrackBars")
#     imgHSV = cv2.cvtColor(imgResize, cv2.COLOR_BGR2HSV)
#     print(h_min,h_max,s_min,s_max,v_min,v_max)
#     lower=np.array([h_min,s_min,v_min])
#     upper=np.array([h_max,s_max,v_max])
#     mask = cv2.inRange(imgHSV,lower,upper)
#     imgResult=cv2.bitwise_and(imgResize,imgResize,mask=mask)
#
#     cv2.imshow("Original", imgResize)
#     #cv2.imshow("HSV", imgHSV)
#     cv2.imshow("Mask",mask)
#     cv2.imshow("image Result",imgResult)
#     cv2.waitKey(1)

############################ Contours / Shape Detecting  #######################
# import cv2
# import numpy as np
# def getContours(img):
#     contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
#     for cnt in contours:
#         area =cv2.contourArea(cnt)
#         print(area)
#
#         if area>200:
#             cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 2)
#             peri=cv2.arcLength(cnt,True)
#             print(peri)
#             approx =cv2.approxPolyDP(cnt,0.02*peri,True )
#             print(len(approx))
#             objCor=len(approx)
#             x,y,w,h= cv2.boundingRect(approx)
#             if objCor == 3: objectType="Tri"
#             elif objCor ==4:
#                 aspRatio =w/float(h)
#                 if aspRatio >0.95 and aspRatio <1.05: objectType = "Square"
#                 else: objectType = "Rectangle"
#             elif objCor>4: objectType="Circles"
#             else:objectType="None"
#             cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
#             cv2.putText(imgContour,objectType,
#                         (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.5,
#                         (0,255,255),2)
#
# img=cv2.imread("Resorce/pic3.jpg")
# imgResize=cv2.resize(img,(400,400))
# imgContour=imgResize.copy()
# imgGray =cv2.cvtColor(imgResize,cv2.COLOR_BGR2GRAY)
# imgBlur=cv2.GaussianBlur(imgGray,(7,7),1)
# imgCanny=cv2.Canny(imgBlur,50,50)
#
#
# cv2.imshow("Original image",imgResize)
# cv2.imshow("Gray image",imgGray)
# cv2.imshow("Blure image",imgBlur)
#
# cv2.imshow("Canny Image",imgCanny)
# getContours(imgCanny)
# cv2.imshow("Ege",imgContour)
# cv2.waitKey(0)

############################################# Face Dectction using cascade #######################


# import cv2
# faceCascade=cv2.CascadeClassifier("Resorce/haarcascade_frontalface_default.xml")
# img=cv2.imread('Resorce/pic.jpg')
# imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# face=faceCascade.detectMultiScale(imgGray,1.1,4)
# for(x,y,w,h) in face:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)
#
# cv2.putText(img,"Face",(350,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
# cv2.imshow("Result",img)
#
#
# cv2.waitKey(0)
############################################ Live webcam face Detection #############################


import cv2
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
minArea=500
color=(255,0,255)
count=0
faceCascade=cv2.CascadeClassifier("Resorce/haarcascade_frontalface_default.xml")
while True:
    success, img = cap.read()

#    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = faceCascade.detectMultiScale(img, 1.1, 4)

    for (x, y, w, h) in face:
        area= w*h
        if area >minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.putText(img,"Face",(x,y-5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,1,color,2)
            imgRoi = img[y:y+h,x:x+w]
            cv2.imshow("ROI",imgRoi)
        cv2.imshow("Result", img)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite("Resorce/Scaned/faces_"+str(count)+".jpg",imgRoi)
            cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
            cv2.putText(img,"Scan Saved",(150,265),cv2.FONT_HERSHEY_DUPLEX,
                        2,(0,0,255),2)
            cv2.imshow("Results",img)
            cv2.waitKey(200)

            count +=1











########################################### Virtual Print #########################################

# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
# cap.set(3,300)
# cap.set(4,300)
# cap.set(10,50)
#
# myColors= [[5,107,0,19,255,255],
#            [133,56,0,159,256,255],
#            [57,76,0,100,255,255]]
# myColorsValus =[[51,153,255],                       ##BGR
#                 [255,0,255],
#                 [0,255,0 ]]
# myPoints= [] ##[x,y,colorId]
#
#
# def finColor(img,myColors,myColorsValus,imgResult ):
#     imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     count=0
#     newPoints=[]
#     for color in myColors:
#         lower = np.array(color[0:3])
#         upper = np.array([color[3:6]])
#         mask = cv2.inRange(imgHSV, lower, upper)
#         x,y=getContours(mask)
#
#         cv2.circle(imgResult,(x,y),10,(myColorsValus[count]),cv2.FILLED)
#         if x!=0 and y!=0:
#             newPoints.append([x,y,count])
#         count +=1
#         #cv2.imshow(str(color[0]), mask)
#     return newPoints
# def getContours(img):
#     contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     x,y,w,h =0,0,0,0
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#
#
#         if area > 200:
#            # cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 2)
#             peri = cv2.arcLength(cnt, True)
#
#             approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
#
#             x, y, w, h = cv2.boundingRect(approx)
#             return x+w//2,y
#
# def drawOnCanvas(myPoints,myColorsValus,imgResult):
#     for points in myPoints:
#         cv2.circle(imgResult, (points[0],points[1]), 10, myColorsValus[points[2]], cv2.FILLED)
#
#
#     while True:
#         success, img = cap.read()
#         imgResult = img.copy()
#         newPoints = finColor(img,myColors,myColorsValus,imgResult)
#         if len(newPoints) !=0:
#             for newP in newPoints:
#                 myPoints.append(newP)
#         if len(myPoints)!=0:
#             drawOnCanvas(myPoints,myColorsValus)
#         cv2.imshow("Result", imgResult)
#         if cv2.waitKey(1) &  0xFF == ord('q'):
#            break

########################################## Document Scanner ##########################

# import cv2
# import numpy as np
# cap = cv2.VideoCapture(0)
# cap.set(3,300)
# cap.set(4,300)
# cap.set(10,50)
#
# def preProcessing():
#
#
#
#
# while True:
#     success, img=cap.read()
#     cv2.imshow("Result", img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
######################################################################################

























































