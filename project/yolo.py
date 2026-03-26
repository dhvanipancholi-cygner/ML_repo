from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt") 

img = cv2.imread("image-dp2.jpeg")

results = model(img)

results[0].show()