import cv2 as cv2
#
video1 = cv2.imread("kizilay.mp4")
# Body Classifier
body_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")
car_cascade = cv2.CascadeClassifier("cars.xml")
# yakalama
body_rect = body_cascade.detectMultiScale(video1)
car_rect = car_cascade.detectMultiScale(video1)

video = cv2.VideoCapture("kizilay.mp4")
def bodydetect(video):
    ret , frame = video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    body_rect = body_cascade.detectMultiScale(gray,minNeighbors = 6)
    car_rect = car_cascade.detectMultiScale(gray,minNeighbors =8)
    for (x, y, w, h) in body_rect:
        cv2.putText(frame, 'insan', (x+10,y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
    for (x, y, w, h) in car_rect:
        cv2.putText(frame, 'araba', (x + 10, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return frame
while True:
    bodydetect(video)
    cv2.imshow("deneme",bodydetect(video))
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cv2.destroyAllWindows()
