import os
import cv2
from typing import List
from ultralytics import YOLO


model = YOLO("yolov5s.pt")
model.train(
    data="data/yaml_dataset/dataset.yaml",
    epochs=20,
)