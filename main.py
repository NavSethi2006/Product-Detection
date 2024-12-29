from ultralytics import YOLO

import cv2
import imutils
import cvzone
import math
import socket
import base64

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO('model/best.pt')

class_names = ["Apple", "Banana", "Grape", "Orange", "Pineapple", "Watermelon"]



serveraddr = ("127.0.0.1", 2000)
buffersize = 1024

UDPClientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)

while True:
    success, img = cap.read()

    img = imutils.resize(img, width=400)
    encoded, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    message = base64.b64encode(buffer)
 
    results = model(img, stream=True)


    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2-x1, y2-y1

            cvzone.cornerRect(img, (x1,y1,w,h))

            conf = math.ceil((box.conf[0]*100))/100

            cls = box.cls[0]

            name = class_names[int(cls)]

            cvzone.putTextRect(img, f'{name}  'f'{conf}', (max(0,x1), max(35,y1)), scale=0.5)

    UDPClientSocket.sendto(message, serveraddr)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        UDPClientSocket.close()
        break

