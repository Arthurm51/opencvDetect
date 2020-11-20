from cv2 import cv2

camera = cv2.VideoCapture(0)
cascadeHand = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
while True:
    _, frame = camera.read()
    frame = cv2.flip(frame,1)
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detectHand = cascadeHand.detectMultiScale(frameGray, 1.2, 5)
    for (x, y, w, h) in detectHand:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame, "Gato detectado", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 0, cv2.LINE_AA)

    cv2.imshow("camera", frame)
    k = cv2.waitKey(60)
    if k == 27:
        break
    
cv2.destroyAllWindows()
camera.release()