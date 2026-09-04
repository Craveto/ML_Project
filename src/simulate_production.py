import mlflow
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
mlflow.set_tracking_uri("databricks")

# 1. Fetch your model directly from Databricks using its unique Run ID
# Copy the Run ID from your Databricks Experiment UI dashboard URL or page

# model_uri = f"runs:/{RUN_ID}/telco_churn_pyfunc_model"

# print(f"Downloading packaged production model from URI: {model_uri}...")
print("model is taking from local path")
loaded_model = mlflow.pyfunc.load_model("C:/ml_project/ml_model/")

# 2. Simulate raw data sent by a frontend production web application (Unprocessed JSON)
raw_production_data = pd.DataFrame([{
    "customerID": "9999-ABCD",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.5,
    "TotalCharges": "845.3"  # Kept as a raw string format on purpose
}])

# 3. Generate prediction
# The custom wrapper processes the string to numbers, runs one-hot encoding, and scores it
prediction = loaded_model.predict(raw_production_data)
print(f"\n🔮 Production Prediction Result: {prediction} (0 = Will Stay, 1 = Will Churn)")
