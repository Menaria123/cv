# Door & Window Detection with YOLOv8

A complete end-to-end pipeline for detecting doors and windows in architectural blueprint images using Ultralytics YOLOv8, with GPU-accelerated training and a Flask-based inference API.

---

## 🔎 Features

- **State-of-the-art object detection:** Leveraging YOLOv8 for high accuracy  
- **Flexible configuration:** Dataset paths and classes defined in `data.yaml`  
- **GPU-accelerated training:** Checkpointing, configurable epochs, batch size, and image size  
- **RESTful inference API:** Deployable Flask app that returns JSON detections  
- **Docker-ready:** Simplify deployment to any container platform  

---
## 🛠️ Setup & Installation

1. **Clone & enter directory**  
   ```bash
   git clone https://github.com/yourusername/door-window-detection.git
   cd door-window-detection
'''create virtual env
   python -m venv venv
   source venv/bin/activate      # macOS/Linux  
   venv\Scripts\activate         # Windows

'''Requirments 
  pip install -r requirements.txt

