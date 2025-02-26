import dagshub
from ultralytics import YOLO
import mlflow
import torch.nn as nn
import logging
from mlflow.tracking import MlflowClient
import yaml
import os
import pandas as pd





# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize YOLO model
model = YOLO(model='yolov3-tinyu.pt')

def mlflow_experiment(yaml_config, model, model_name, experiment_name):
    # Validate YAML file
    if not yaml_config or not yaml_config.endswith('.yaml'):
        logger.error("❌ Config file path must be a valid .yaml file")
        return

    # Load YAML config
    try:
        with open(yaml_config, 'r') as config_file:
            config = yaml.safe_load(config_file)
    except FileNotFoundError:
        logger.error(f"❌ Config file not found: {yaml_config}")
        return
    except yaml.YAMLError as e:
        logger.error(f"❌ YAML parsing error: {e}")
        return

    # Validate model: It has to be a pytorch model instance
    if not model or not isinstance(model, (nn.Module, YOLO)):
        logger.error("❌ Model field empty or not of type nn.Module or YOLO")
        return

    # Validate experiment and model names
    if not experiment_name:
        logger.error("❌ Must specify experiment name.")
        return

    if not model_name:
        logger.error("❌ Must specify model name.")
        return

    #Set up experiment name
    try:
        mlflow.set_experiment(experiment_name=experiment_name)
        logger.info(f"✅ Experiment '{experiment_name}' created.")
    except Exception as e:
        logger.error(f"❌ Failed to set experiment: {e}")
        return
    
    try:
        with mlflow.start_run(run_name = experiment_name) as run:
            run_id = run.info.run_id
            print(f"🚀 MLflow Run ID: {run_id}")
            
            #Log parameters
            mlflow.log_param("epochs",50)
            mlflow.log_param("batch_size",16)
            mlflow.log_param("imgsz",640)
            mlflow.log_param("learning_rate",0.001)
            mlflow.log_param("optimizer","Adam")
            
            #Train model
            results = model.train(
            data=yaml_config,
            epochs=100,
            batch=8,
            imgsz=224,
            lr0=0.01,
            optimizer='Adam',
            freeze = 5,
            degrees=5.0,       # very slight rotation, ±1 degree
            translate=0.02,
            hsv_h=0.005,
            hsv_s=0.2,         
            hsv_v=0.2          

        )   
            #Results df
            run_dir = results.save_dir 
            results_dir = os.path.join(run_dir,"results.csv")
            mlflow.log_artifact(results_dir) #Log results

            results_df = pd.read_csv(results_dir)
            #Get results of best epoch
            best_epoch = results_df.loc[results_df['metrics/mAP50-95(B)'].idxmax()]
            precision = best_epoch['metrics/precision(B)']
            recall = best_epoch['metrics/recall(B)']
            mAP_0_5 = best_epoch['metrics/mAP50(B)']
            mAP_0_5_95 = best_epoch['metrics/mAP50-95(B)']
            #Calculate f1 score
            if precision + recall > 0:
                f1_score = 2* (precision*recall)/(precision+recall)
            else:
                f1_score = 0 
            
            #Log metrics
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("F1_Score", f1_score)
            mlflow.log_metric("mAP_0.5", mAP_0_5)
            mlflow.log_metric("mAP_0_5_0_95", mAP_0_5_95)

            #Log model

            best_model_path = os.path.join(run_dir, "weights", "best.pt")
            if os.path.exists(best_model_path):
                mlflow.log_artifact(best_model_path, artifact_path="YOLO_Model")
                logger.info(f"✅ Logged trained YOLO model: {best_model_path}")
            else:
                logger.warning("⚠️ Best model file not found, skipping artifact logging.")
            
            mlflow.end_run()

    except Exception as e:
        logger.error(f"❌ Error during MLflow run: {e}")

 
'''
if  __name__ == "__main__":
    model = YOLO(model='yolov3-tinyu.pt') 
    mlflow_experiment('model/yolo_v3_mini/yolo_v3_mini.yaml',
                      model,
                      "yolo_v3_mini",
                      "Resize_pad_v2")

'''
 #Run this in terminal

    #python -m model.yolo_v3_mini.yolo_v3_model   