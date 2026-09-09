import os
import pandas as pd
import numpy as np
import mlflow
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient

# Evidently AI Imports for Monitoring
from evidently import Report
from evidently.presets  import DataDriftPreset

# metric_preset
# 1. Initialize Environments and Databricks Connection
load_dotenv()
mlflow.set_tracking_uri("databricks")

# --- CONFIGURATION (Update these variables) ---
RUN_ID = "f7f50828ac4e48f5bcb7d9a29f9a8cf6"  # Copy this from your Databricks Experiment UI
USER_EMAIL = "snehaldighore87@gmail.com"
VOLUME_CSV_PATH = "/Volumes/aiml/data/churn_data/WA_Fn-UseC_-Telco-Customer-Churn.csv" 
# ----------------------------------------------

# 2. Load the Packaged Custom PyFunc Model from Databricks
model_uri = f"runs://{RUN_ID}/telco_churn_pyfunc_model"
print(f"📥 Loading custom production model from MLflow: {model_uri}")
model = mlflow.pyfunc.load_model("C:/ml_project/ml_model/")


# 3. Fetch Baseline Reference Data (Original Dataset)
print("📥 Fetching baseline reference dataset for drift comparison...")
w = WorkspaceClient()
local_raw_path = "C:/ml_project/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# with w.files.download(VOLUME_CSV_PATH) as response:
#     with open(local_raw_path, "wb") as f:
#         f.write(response.contents.read())

reference_df = pd.read_csv(local_raw_path)

# 4. Simulate a Fresh Batch of Production Data with Synthetic "Drift"
print("🧪 Simulating a new batch of incoming production data with behavioral drift...")
production_df = reference_df.copy().sample(n=1000, random_state=101).reset_index(drop=True)

# Artificially introduce drift to see if our monitor catches it:
# Shift customers to short term contracts and spike their charges
production_df["Contract"] = "Month-to-month"
production_df["MonthlyCharges"] = production_df["MonthlyCharges"] * 1.5 
production_df["TotalCharges"] = production_df["TotalCharges"].replace(" ", np.nan)
production_df["TotalCharges"] = pd.to_numeric(production_df["TotalCharges"].fillna(0)) * 1.5
production_df["TotalCharges"] = production_df["TotalCharges"].astype(str)

# 5. Run Batch Scoring (Accepts completely raw un-engineered data!)
print("🔮 Running Batch Scoring on production data...")
# We drop the actual target 'Churn' if it's there, simulating real production inputs
raw_features_prod = production_df.drop(columns=["Churn"], errors="ignore")
production_df["Predictions"] = model.predict(raw_features_prod)

print(f"✅ Batch scoring complete! Sample predictions: {list(production_df['Predictions'].head())}")

# 6. Run Data Drift Monitoring using Evidently AI
print("📊 Analyzing Data Drift between training baseline and production batch...")

# Clean up target and ID columns from monitoring inputs so they don't skew feature analysis
features_to_monitor = [col for col in reference_df.columns if col not in ["customerID", "Churn"]]

drift_report = Report(metrics=[
    DataDriftPreset()
])

# Evidently compares the distribution of features between baseline and production
snapshot = drift_report.run(
    reference_data=reference_df[features_to_monitor], 
    current_data=production_df[features_to_monitor]
)

# Save the dashboard as an interactive webpage html file
report_html_path = "data_drift_report.html"
snapshot.save_html(report_html_path)
print(f"🎉 Monitoring Complete! HTML report generated at: {report_html_path}")

# Clean up local raw file
if os.path.exists(local_raw_path):
    os.remove(local_raw_path)
