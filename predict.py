import os
import cv2
from typing import List
from ultralytics import YOLO
from ultralytics import YOLO


model = YOLO("best.pt")

def load_data_images(source_dir: str) -> List[str]:
    """Recursively load all images in a directory"""

    data_images = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".png"):
                data_images.append(os.path.join(root, file))
    return data_images

def split_images_into_target_no_target(data_images: List[str]) -> List[str]:
    """Split the list of images into two lists, one with images that have a target, one without"""

    target_images = []
    no_target_images = []
    for image in data_images:
        if "no_targets" in image:
            no_target_images.append(image)
        elif "target" in image:
            target_images.append(image)
        else:
            raise ValueError("Image does not have a target or no_target label")
    return target_images, no_target_images

example_images = load_data_images("data/processed_frames")
target_images, no_target_images = split_images_into_target_no_target(example_images)
example_images = target_images

# create window to show images in
window = cv2.namedWindow("image", cv2.WINDOW_NORMAL)

for image in example_images:
    # resize image to 640x640
    img = cv2.imread(image)
    img = cv2.resize(img, (640, 640))

    # run model on image
    results = model(img)
    print(f"Found {len(results)} objects in image {image} the results are: {results}")

    # show image in window
    cv2.imshow("image", img)

    # draw bounding boxes
    for result in results:
        boxes = result.boxes
        for box in boxes:

            data = box.xyxy
            print(f"Bounding box: {data}")
            x1, y1, x2, y2 = data[0]

            # convert to int
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("image", img)


    cv2.waitKey(0)



