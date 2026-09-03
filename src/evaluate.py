import mlflow
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.custom_model import TelcoChurnPyFuncModel

def evaluate_best_model_and_log_custom_model(best_params, X_train, X_val, y_train, y_val):
    print("📊 Evaluating ultimate champion model configuration...")
    
    if best_params["model_type"] == "xgboost":
        model = xgb.XGBClassifier(
            learning_rate=best_params.get("xgb_lr", 0.1),
            max_depth=best_params.get("xgb_depth", 5),
            n_estimators=100
        )
    else:
        model = RandomForestClassifier(
            n_estimators=best_params.get("rf_estimators", 100),
            max_depth=best_params.get("rf_depth", 10)
        )
        
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]
    
    # Calculate performance metrics
    metrics = {
        "val_accuracy": accuracy_score(y_val, preds),
        "val_precision": precision_score(y_val, preds),
        "val_recall": recall_score(y_val, preds),
        "val_f1": f1_score(y_val, preds),
        "val_roc_auc": roc_auc_score(y_val, probs)
    }
    
    # Log metrics to parent execution
    mlflow.log_metrics(metrics)
    
    
     # Get the exact list of columns the model expects
    training_columns = list(X_train.columns)
    
    # Instantiate the Custom PyFunc wrapper
    pyfunc_wrapper = TelcoChurnPyFuncModel(
        trained_model=model,
        training_columns=training_columns
    )
    
    # 📦 Log the Custom PyFunc Model to MLflow Tracking Server
    print("📦 Uploading integrated Custom PyFunc Model to Databricks...")
    mlflow.pyfunc.log_model(
        artifact_path="telco_churn_pyfunc_model",
        python_model=pyfunc_wrapper
    )
    print("✅ Model logged successfully!")
    
    # print(f"Champion Validation Metrics logged: {metrics}")
