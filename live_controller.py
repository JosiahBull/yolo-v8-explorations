from multiprocessing import Queue
import os
from threading import Thread
from mss import mss
import cv2
from typing import List
from ultralytics import YOLO
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

save_threshold = 0.3
save_directory = "data/uncertain_predictions"

def saver_thread(queue):
    """Save images from queue to disk"""
    while True:
        # get the next image from the queue
        img, filename = queue.get()

        # save the image
        image = Image.fromarray(img[:, :, :3])
        image.save(os.path.join(save_directory, filename))
        print("Saved image")

        # mark the task as done
        queue.task_done()


sct = mss()

model = YOLO("best.pt")

ms_per_frame = 1000 / 60

# create window to show images in
window = cv2.namedWindow("image", cv2.WINDOW_NORMAL)

# create queue to save images to
queue = Queue()
saver = Thread(target=saver_thread, args=(queue,))


while True:
    # capture current time
    now = time.time()

    # capture image of primary screen
    img_original = np.array(sct.grab(sct.monitors[1]))

    # scale image to 640x640
    img = cv2.resize(img_original, (640, 640))

    # run model on image
    results = model(img[:, :, :3])
    # print(f"Found {len(results)} objects in image the results are: {results}")

    # draw bounding boxes
    for result in results:
        boxes = result.boxes

        save_img = False

        for box in boxes:

            if box.conf < save_threshold:
                save_img = True

            data = box.xyxy
            # print(f"Bounding box: {data}")
            x1, y1, x2, y2 = data[0]

            # convert to int
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if save_img:
            # save image
            queue.put((img_original, f"{time.time()}.png"))

    cv2.imshow("image", img)

    # wait for the next frame
    wait_time = max(0, ms_per_frame - (time.time() - now))
    cv2.waitKey(int(wait_time))