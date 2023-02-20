

import shutil
import json
import os
from typing import List


source_dir = "data/processed_frames"
dataset_output_dir = "data/yaml_dataset"

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


images = load_data_images(source_dir)
target_images, no_target_images = split_images_into_target_no_target(images)

labelled_images = []
for image in target_images:
    if os.path.exists(image.replace(".png", ".json")):
        labelled_images.append(image)

print("Number of images with targets: ", len(target_images))

# create dataset output dir if not exist
if not os.path.exists(dataset_output_dir):
    os.makedirs(dataset_output_dir)

# create yaml file
with open(os.path.join(dataset_output_dir, "dataset.yaml"), "w") as f:
    f.write("path: ../hit-detector/data/yaml_dataset\n" )
    f.write("train: images\n" )
    f.write("val: images\n" )
    # f.write("test: " )

    f.write("names:\n")
    f.write("  0: Enemy\n")

# copy files into dataset dir
if not os.path.exists(os.path.join(dataset_output_dir, "images")):
    os.makedirs(os.path.join(dataset_output_dir, "images"))
for image in labelled_images:
    parent_directory_name = os.path.basename(os.path.dirname(image))
    second_parent_directory_name = os.path.basename(os.path.dirname(os.path.dirname(image)))


    output_name = second_parent_directory_name + "_" + parent_directory_name + "_" + os.path.basename(image)
    shutil.copy(image, os.path.join(dataset_output_dir, "images", output_name))

    # resize image to be 640x640
    import cv2
    img = cv2.imread(os.path.join(dataset_output_dir, "images", output_name))
    img = cv2.resize(img, (640, 640))
    cv2.imwrite(os.path.join(dataset_output_dir, "images", output_name), img)


# create labels dir
if not os.path.exists(os.path.join(dataset_output_dir, "labels")):
    os.makedirs(os.path.join(dataset_output_dir, "labels"))
for image in labelled_images:
    # save label to as a .txt file with the correct name
    json_data = json.loads(open(image.replace(".png", ".json")).read())

    parent_directory_name = os.path.basename(os.path.dirname(image))
    second_parent_directory_name = os.path.basename(os.path.dirname(os.path.dirname(image)))
    filename = second_parent_directory_name + "_" + parent_directory_name + "_" + os.path.basename(image).replace(".png", ".txt")


    labels = json_data["bounding_boxes"]

    # normalize labels to be within 0 -> 1
    for label in labels:
        # label 0 is x (0 -> 2560)
        # label 1 is y (0 -> 1440)
        # label 2 is width (0 -> 2560)
        # label 3 is height (0 -> 1440)

        # convert label_0 to be x_center
        label[0] = label[0] + (label[2] / 2)
        # convert label_1 to be y_center
        label[1] = label[1] + (label[3] / 2)

        # normalize the values to be within 0 -> 1
        label[0] = label[0] / 2560
        label[1] = label[1] / 1440
        label[2] = label[2] / 2560
        label[3] = label[3] / 1440

    with open(os.path.join(dataset_output_dir, "labels", filename), "w") as f:
        for label in labels:
            # write label to file
            f.write("0 {} {} {} {}\n".format(label[0], label[1], label[2], label[3]))

