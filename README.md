Helmet Detection Using YOLOv11
Project Overview

This project focuses on detecting whether a motorcycle rider is wearing a helmet or not using a deep learning object detection model. The system is designed to support traffic safety enforcement and accident prevention by automatically identifying helmet usage from images.

The project is built using the YOLOv11 object detection framework and trained on a labeled helmet detection dataset.

Objective

The main objective of this project is to automatically detect helmet usage in images of motorcycle riders. The system classifies riders into two categories:

Helmet

No Helmet

Dataset Description

The dataset contains images of motorcycle riders captured under different traffic and environmental conditions.

Each image is annotated using bounding boxes in YOLO format.

Classes included:

Helmet

No Helmet

The dataset is split into:

Training set

Validation set

Test set

Methodology

Collected and prepared helmet and no-helmet images

Annotated images using bounding boxes

Converted annotations into YOLO format

Used transfer learning with YOLOv11 pretrained weights

Trained the model on the helmet detection dataset

Evaluated model performance on validation data

Performed inference on new images to verify results

Model Used

YOLOv11 Nano (yolo11n.pt)

Chosen for fast inference and good accuracy

Pretrained on COCO dataset and fine-tuned for helmet detection

Training Details

Image size: 640 × 640

Batch size: 16

Epochs: 25

Framework: Ultralytics YOLO

Hardware: NVIDIA Tesla T4 (Google Colab)

Results

The trained model successfully detects helmets with high confidence.

Bounding boxes are accurately drawn around helmets.

Inference speed is fast, making the model suitable for real-time applications.

Prediction results are automatically saved for review.