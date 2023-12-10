import cv2
import numpy
import imutils

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
scaling_factor = 0.5
frame = cv2.imread("durov.jpg")
frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
for(x, y, w, h) in face_rects:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 3)

cv2.imshow("durov.jpg", frame)
cv2.waitKey(0)
print(f'Found {len(face_rects)} faces!')

smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

image = cv2.imread("durov.jpg")
gray_filter = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_filter, 7, 4)

for(x, y, w, h) in faces:
    cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 2)
    roi_gray = gray_filter[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    smile = smile_cascade.detectMultiScale(roi_gray)
    eye = eye_cascade.detectMultiScale(roi_gray)
    for(sx, sy, sw, sh) in smile:
        cv2.rectangle(roi_color, (sx, sy), (sx+w, sy+h), (0,255,0), 1)
    for (ex, ey, ew, eh) in eye:
        cv2.rectangle(roi_color, (ex, ey), (ex + w, ey + h), (0, 0, 255), 1)

cv2.imshow("Image", image)
cv2.waitKey(0)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
image = cv2.imread("screenshot.png")
image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
people_rects = hog.detectMultiScale(image, winStride=(8,8), padding=(30,30), scale=1.06)

for(x,y,w,h) in people_rects[0]:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("X--men", image)
cv2.waitKey(0)
print(f'Found {len(people_rects[0])} people!')



cv2.startWindowThread()
cap = cv2.VideoCapture('video_with_people.mp4')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 560))
    gray_filter = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))
    boxes = numpy.array([[x, y, x+w, y+h] for (x,y,w,h) in boxes])

    for(xa, ya, xb, yb) in boxes:
        cv2.rectangle(frame, (xa, ya), (xb, yb), (0,255,0), 1)
    cv2.imshow("Video", frame)
    if(cv2.waitKey(1) & 0XFF==ord('q')):
        break

cap.release()
cv2.destroyAllWindows()

