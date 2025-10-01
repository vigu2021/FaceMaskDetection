# FaceMaskDetection
AI-powered face mask detection system using deep learning 
# Mask Detection API 🚀

This is a FastAPI-based mask detection system that uses a deep learning model to determine whether a person is wearing a mask.

## 🖼️ Example Images
| Without Mask | With Mask |
|-------------|----------|
| ![Without Mask](Screenshot%202025-02-26%20173017.png) | ![With Mask](Screenshot%202025-02-26%20173145.png) |


## 📌 **Features**
**Real-time mask detection** via webcam  
**Image upload support** for mask classification  
**Bounding boxes with labels** (Mask / No Mask)  
**Confidence scores displayed**  

## 🔧 **Technologies Used**
 **FastAPI** - Backend framework for handling API requests
 **YOLOv3 mini** - Object detection model for face mask classification
 **OpenCV** - Capturing and processing video frames
 **Docker** - Containerization for easy deployment  

## 🎯 **How It Works**
1. **Upload an Image** 📤 or **Use Live Webcam** 🎥
2. The **FastAPI backend** processes the image/video frame
3. **YOLOv3 mini model** detects if a person is **wearing a mask** or **not**
4. The bounding box and confidence score are displayed 🔲✅❌  

## 🛠 **Installation & Setup**
### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/your-username/mask-detection-app.git
cd mask-detection-app
