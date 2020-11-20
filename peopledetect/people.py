from imutils.object_detection import non_max_suppression
import imutils
import cv2
import numpy as np


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cam = cv2.VideoCapture(0)
while True:
    retval, image = cam.read()
    image = cv2.flip(image, 1)
    image = imutils.resize(image, width=min(600, image.shape[1]))
    orig = image.copy()
    (rects, weights) = hog.detectMultiScale(
        image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.putText(image, "Pessoas Detectadas: ", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 0, cv2.LINE_AA)
    cv2.imshow("After NMS", image)
    k = cv2.waitKey(1)
    if k == 27:
        break


cv2.destroyAllWindows()
