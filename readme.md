# YOLOV8 Explorations

Decided to try out a quick test project to see how well YoloV8 performs. This is a simple application of the machine learning tool which aims to recognize targets in the video game "RoboCraft" - A domain I'm familiar with.

I recorded ~30 minutes of gameplay, split it into frames, labelled around 1000 of them and then ran it on some unseen videos. The AI performed extremely well, and was able to track objects across multiple frames with ease.

## Scripts
- Rust Program - Splits video into frames, and then filters out frames which are not of interest - e.g. missing certain pixel colors that indicate the precense oof target usernames.
- label.py - As the name suggests, this creates a small window where a user can quickly label random frames of interest.
- package_data.py - Converts the labelled frames into a format that can be used by YoloV8.
- train.py - Trains the model using the labelled data.
- predict.py - Runs the model on frames from the dataset.=
- live_controller.py - Runs the model on frames from a live video feed.