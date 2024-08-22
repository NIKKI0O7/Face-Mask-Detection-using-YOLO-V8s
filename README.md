# Face-Mask-Detection-using-YOLO-V8s
Face Mask Detection using YOLO-V8


## Overview

This project involves training and deploying a YOLOv8 object detection model. The pipeline includes data preparation, model training, and video processing. 

## Table of Contents

- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Video Processing](#video-processing)
- [Usage](#usage)
- [License](#license)

## Project Structure

## Dependencies

Ensure you have the following dependencies installed:

- Python 3.x
- OpenCV
- Pandas
- Matplotlib
- Scikit-learn
- Ultralytics YOLO

You can install the necessary packages using:

```bash
pip install opencv-python-headless pandas matplotlib scikit-learn ultralytics

Data Preparation
Data Structure
Ensure your data directory contains images and label files in YOLO format.

Label Format
Convert annotation data to YOLO format. Organize images and labels into training and validation directories.

Splitting Data
Use the provided script to split your data into training and validation sets:

python
Copy code
from sklearn.model_selection import train_test_split

# Split dataframe into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
Model Training
Training Script
Run train.py to train the YOLOv8 model. Update the data.yaml file with paths to your training and validation data:

python
Copy code
from ultralytics import YOLO

model = YOLO("yolov9e.pt")
results = model.train(data="data.yaml", epochs=200, device=[0], batch=32)
Configuration
Ensure that data.yaml contains the correct paths to the training and validation datasets.

Video Processing
Processing Script
Use process_video.py to process a video file with the trained YOLOv8 model. The script will read the video, perform object detection, and save the output with bounding boxes and labels:

python
Copy code
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov9e.pt")

video_path = 'Test_video1.mp4'
output_path = 'output_video1_v2.mp4'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    for result in results:
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_ids = result.boxes.cls

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[int(class_id)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
print("Video processing complete. Output saved to", output_path)
Input/Output
Specify the input video file path and desired output path in the script.

Usage
To train the model, run:

bash
Copy code
python train.py
To process a video, run:

bash
Copy code
python process_video.py
License
This project is licensed under the MIT License. See the LICENSE file for details.

csharp
Copy code

You can copy and paste this directly into your `README.md` file for GitHub.

