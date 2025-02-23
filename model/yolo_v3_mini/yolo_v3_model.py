from ultralytics import YOLO
import mlflow as ml


model = YOLO(model='yolov3-tinyu.pt') 
print(model.children,model.modules)