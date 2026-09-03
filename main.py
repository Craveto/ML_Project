import os
import yaml
import mlflow
from dotenv import load_dotenv
from src.data_ingestion import download_data_from_volume
from src.feature_engineering import preprocess_telco_data
from src.train import train_and_select_model
from src.evaluate import evaluate_best_model_and_log_custom_model

load_dotenv()

# Read Configuration parameters
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Connect Tracking URL
mlflow.set_tracking_uri("databricks")
USER_EMAIL = os.getenv("DATABRICKS_USER_EMAIL", "your_email@example.com") 
mlflow.set_experiment(config['mlflow']['experiment_name_template'].format(USER_EMAIL))

# Exact path details matching your environment setup configuration
VOLUME_CSV_PATH = os.getenv("VOLUME_CSV_PATH","/Volumes/aiml/data/your_volumePath/WA_Fn-UseC_-Telco-Customer-Churn.csv")

def run_pipeline():
    with mlflow.start_run(run_name=config['mlflow']['parent_run_name']):
        
        # Phase 1 & 2: Ingest and Engineer Features
        local_file = download_data_from_volume(VOLUME_CSV_PATH)
        X, y = preprocess_telco_data(local_file, config['data']['target_column'], config['data']['id_column'])
        
        # Phase 3: Train and Select Top Framework Model Architecture
        best_params, X_train, X_val, y_train, y_val = train_and_select_model(X, y, config)
        mlflow.log_params(best_params)
        
        # Phase 4: Final Validation Evaluation
        evaluate_best_model_and_log_custom_model(best_params, X_train, X_val, y_train, y_val)
        
        print("\n🎉 End-to-End MLOps Pipeline Run execution completed successfully!")

if __name__ == "__main__":
    run_pipeline()
