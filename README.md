# Door and Window Detection with YOLOv8

A simple Flask-based REST API for detecting doors and windows in images using a custom-trained YOLOv8 model. This repository includes scripts for data preparation, model training, and inference (via a Flask server).

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Directory Structure](#directory-structure)  
- [Requirements](#requirements)  
- [Setup & Installation](#setup--installation)  
- [Data Organization](#data-organization)  
- [Training the Model](#training-the-model)  
- [Running the Inference Server](#running-the-inference-server)  
- [API Usage](#api-usage)  
- [Examples](#examples)  
- [Notes & Tips](#notes--tips)  
- [License](#license)

---

## Project Overview

This project demonstrates how to train a custom YOLOv8 model to detect doors and windows and how to serve detection results via a Flask REST API. The main components are:

1. **Data Preparation**: Organize images and labels in YOLO format.  
2. **Training Script**: A Python script (`train.py`) to train YOLOv8 on the door/window dataset.  
3. **Inference Server**: A Flask app (`app.py`) that loads the trained weights and exposes a `/detect` endpoint.  
4. **Example Client**: Shows how to call the endpoint using `curl` or any HTTP client.

---

## Directory Structure
![image](https://github.com/user-attachments/assets/dfab8d4f-7038-4b33-8b43-f538c16773af)


- **images/train, images/val**: Training and validation images.  
- **labels/train, labels/val**: Corresponding YOLO-format `.txt` annotation files.  
- **data.yaml**: Dataset configuration file for YOLOv8.  
- **requirements.txt**: Python dependencies.  
- **app.py**: Flask application serving detection results.  
- **train.py**: Training script for the custom YOLOv8 model.  
- **yolov8n.pt**: Pretrained YOLOv8n weights (downloaded from Ultralytics).  
- **runs/train/door_window_yolov8/weights/best.pt**: Trained weights output.

---

## Setup & Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/door-window-detection.git
   cd door-window-detection
   
2. **Create a Python Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate         # Windows
   
3.**Install Dependencies**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt

4.**How to run**
   ```bash 
    python train.py
    python app.py
