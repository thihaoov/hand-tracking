import cv2
import numpy as np
import time
import os
import HandTrackingModule as hdm

#######################################
brushThickness = 5
eraserThickness = 30
#######################################

folderPath = "Header"
myList = os.listdir(folderPath)
myList.sort()
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
# print(len(overlayList))

header = overlayList[0]

drawColor = (0,0,255)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 360)

detector = hdm.handDetector(maxHands=1)
xp, yp = 0, 0 # x previous and y pre
imgCanva = np.zeros((360, 640, 3), np.uint8)
# imgCanva.fill(255)

while True:

    # 1. Import IMage
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        # print(lmList)

        #  tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

    # 4. If selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            print("Secletion Mode")
            xp, yp = 0, 0
            # Checking for the click
            if(y1 < 62):
                if 135 < x1 < 190:
                    header = overlayList[0]
                    drawColor = (0,0,255)
                if 265 < x1 < 315:
                    header = overlayList[1]
                    drawColor = (255,0,0)
                if 405 < x1 < 450:
                    header = overlayList[2]
                    drawColor = (0,200,0)
                if 535 < x1 < 585:
                    header = overlayList[3]
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1,y1-15), (x2,y2+15), drawColor, cv2.FILLED)
        

    # 5. If Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == 0:
            cv2.circle(img, (x1, y1), 10, drawColor, cv2.FILLED)
            print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp,yp), (x1,y1), drawColor, eraserThickness)
                cv2.line(imgCanva, (xp,yp), (x1,y1), drawColor, eraserThickness) 
            else:
                cv2.line(img, (xp,yp), (x1,y1), drawColor, brushThickness)
                cv2.line(imgCanva, (xp,yp), (x1,y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanva, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img, imgCanva)

    # Setting Header Image
    img[0:62, 0:640] = header
    img = cv2.addWeighted(img, 0.5, imgCanva, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canva", imgCanva)
    cv2.imshow("Inverse", imgInv)
    cv2.waitKey(1)
