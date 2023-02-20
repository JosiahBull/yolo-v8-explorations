import os
from typing import List


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

for image in target_images:
    output_file = image.replace(".png", ".json")
    if not os.path.exists(output_file):
        # remove the image
        os.remove(image)