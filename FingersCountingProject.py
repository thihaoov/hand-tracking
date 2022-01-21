import cv2
import time
import os
import HandTrackingModule as htm

###################################
wCam, hCam = 640, 480
###################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
myList.sort()
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)

pTime = 0

detector = htm.handDetector(maxHands=1, detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList[8][2], lmList[6][2])
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else: fingers.append(0)
        
        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else: fingers.append(0)

        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers].shape
        img[0:h, 0:w] = overlayList[totalFingers]

        cv2.rectangle(img, (20,300), (170,500), (0,255,0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45,435), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(
        img, f"FPS: {int(fps)}", (480, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2
    )
    cv2.imshow("Finger", img)
    cv2.waitKey(1)
