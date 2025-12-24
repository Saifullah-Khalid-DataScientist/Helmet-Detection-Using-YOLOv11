🪖 Helmet Detection Using YOLOv11
📌 Project Overview

This project focuses on automatically detecting whether a motorcycle rider is wearing a helmet or not using a deep learning–based object detection model.
The system is designed to support traffic safety enforcement, accident prevention, and smart surveillance systems.

The model is trained using YOLOv11, which provides fast and accurate real-time object detection.

🎯 Objective

The main objective of this project is to:

Detect helmet and no-helmet cases from traffic images

Help authorities enforce helmet safety rules

Reduce road accidents through automated monitoring

📂 Dataset Description

The dataset contains images of motorcycle riders captured in various traffic conditions.

Classes:

Helmet

No Helmet

Dataset Features:

Different camera angles

Daytime and nighttime images

Multiple traffic environments

YOLO format annotations

🛠️ Methodology

Collected motorcycle rider images

Annotated helmet and no-helmet regions using bounding boxes

Converted annotations into YOLO format

Trained YOLOv11 using transfer learning

Fine-tuned the model to improve detection accuracy

Tested the model on unseen images

🚀 Model & Training

Model Used: YOLOv11 Nano (yolo11n.pt)

Framework: Ultralytics YOLO

Image Size: 640 × 640

Training Epochs: Configurable

Hardware: GPU (Tesla T4 on Google Colab)

📊 Results

The trained model successfully:

Detects helmets with high confidence

Draws accurate bounding boxes

Works efficiently on real-world images

Example output includes bounding boxes with confidence scores displayed on detected helmets.


model = YOLO("best.pt")
model.predict(
    source="test_image.jp
