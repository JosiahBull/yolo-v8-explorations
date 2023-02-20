# OK this program is a machine learning experiment
# The goal is to load a bunch of unlabelled data, feed it to a Neural Network
# for each data point, the NN will make a guess on the localisation of an object using something like YOLO
# We then show the result to a human (me) in a window
# if we are happy, I left click and the data point/label is saved
# if we are not happy, I right click on the image to draw labels until a left click is made

import glob
import json
import os
import cv2
from typing import List
from ultralytics import YOLO

source_dir = "data/processed_frames"

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



data_images = load_data_images(source_dir)
target_images, no_target_images = split_images_into_target_no_target(data_images)

print("Number of images with targets: ", len(target_images))
print("Number of images without targets: ", len(no_target_images))

# create a window to show the images in
# window = cv2.namedWindow("image", cv2.WINDOW_NORMAL)

# loop and find each image that actually has a json file
counter = 0
for image in target_images:
    output_file = image.replace(".png", ".json")
    if os.path.exists(output_file):
        counter += 1
        # show this image in the window
        # img = cv2.imread(image)
        # cv2.imshow("image", img)

        # # read json file
        # data = json.load(open(output_file))
        # bounding_boxes = data["bounding_boxes"]

        # # draw bounding boxes
        # for bounding_box in bounding_boxes:
        #     cv2.rectangle(img, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 255, 0), 2)
        #     cv2.imshow("image", img)

        # # if the user presses space, keep the json file
        # # if the user presses enter, delete the json file
        # # if the user presses esc, exit the program
        # key = cv2.waitKey(0)
        # if key == 27:
        #     # ESC key
        #     break
        # elif key == 13:
        #     # enter key
        #     os.remove(output_file)
        #     print("Deleted json file: ", output_file)
        # elif key == 32:
        #     # space key
        #     print("Keeping json file: ", output_file)
        #     counter += 1


# randomly shuffle the images
import random
random.shuffle(target_images)

# for each image, show it in the window and allow the user to draw a bounding box
for i, image in enumerate(target_images):
    output_file = image.replace(".png", ".json")

    # check if json file already exists
    if os.path.exists(output_file):
        print("Skipping image, json file already exists: ", image)
        continue

    # show image in window
    print("Showing image: ", image)
    img = cv2.imread(image)
    cv2.imshow("image", img)

    # display the counter in the upper right corner
    cv2.putText(img, str(counter), (img.shape[1] - 400, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 2)
    cv2.imshow("image", img)

    # add instructions
    cv2.putText(img, "Press 1 to draw enemy robot bounding box", (img.shape[1] - 400, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, "Press 2 to draw ally robot bounding box", (img.shape[1] - 400, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, "Press 3 to draw enemy base bounding box", (img.shape[1] - 400, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # wait for user to drag a bounding box
    # when the user is happy, they left click

    enemy_robot_bounding_boxes = []
    ally_robot_bounding_boxes = []
    enemy_base_bounding_box = []

    running = True
    while running:
        # wait for user to click
        key = cv2.waitKey(0)
        if key == 27:
            # ESC key - exit program
            running = False
            break
        elif key == 13:
            # enter key
            break
        elif key == 32:
            # space key
            break

        # 1 key = create enemy bounding box, or B
        elif key == 49 or key == 98:
            # wait for user to drag bounding box
            bounding_box = cv2.selectROI("image", img, fromCenter=False, showCrosshair=True)
            enemy_robot_bounding_boxes.append(bounding_box)
            print("Bounding box: ", bounding_box)
            # draw RED bounding box on image
            cv2.rectangle(img, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 0, 255), 2)
            cv2.imshow("image", img)

        # 2 key = create ally bounding box
        elif key == 50:
            # wait for user to drag bounding box
            bounding_box = cv2.selectROI("image", img, fromCenter=False, showCrosshair=True)
            ally_robot_bounding_boxes.append(bounding_box)
            print("Bounding box: ", bounding_box)
            # draw GREEN bounding box on image
            cv2.rectangle(img, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 255, 0), 2)
            cv2.imshow("image", img)

        # 3 key = create enemy base bounding box
        elif key == 51:
            # wait for user to drag bounding box
            bounding_box = cv2.selectROI("image", img, fromCenter=False, showCrosshair=True)
            enemy_base_bounding_box = bounding_box
            print("Bounding box: ", bounding_box)
            # draw BLUE bounding box on image
            cv2.rectangle(img, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (255, 0, 0), 2)
            cv2.imshow("image", img)

        # c key = clear bounding boxes
        elif key == 99:
            bounding_boxes = []
            # clear rectangles from image
            img = cv2.imread(image)
            cv2.imshow("image", img)
        # z key = delete the last file's bounding boxes
        elif key == 122:
            target_image = target_images[i - 1]
            output_file = target_image.replace(".png", ".json")
            if os.path.exists(output_file):
                os.remove(output_file)
                print("Deleted json file: ", output_file)
        else:
            print("Key pressed: ", key)

    if not running:
        break

    # save bounding boxes to file
    output = {
        "image": image,
        "enemy_robot_bounding_boxes": enemy_robot_bounding_boxes,
        "ally_robot_bounding_boxes": ally_robot_bounding_boxes,
        "enemy_base_bounding_box": enemy_base_bounding_box
    }

    with open(output_file, "w") as f:
        json.dump(output, f)
    print("Saved bounding boxes to file: ", output_file)
    counter += 1
