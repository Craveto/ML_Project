import optuna
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import mlflow
from sklearn.model_selection import train_test_split

def train_and_select_model(X, y, config):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config['data']['test_size'], 
        random_state=config['data']['random_state'],
        stratify=y
    )

    def objective(trial):
        # Model Selection Space: XGBoost vs RandomForest
        model_type = trial.suggest_categorical("model_type", ["xgboost", "random_forest"])
        
        with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number}_{model_type}"):
            if model_type == "xgboost":
                mlflow.xgboost.autolog(log_models=True)
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "logloss",
                    "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.2, log=True),
                    "max_depth": trial.suggest_int("xgb_depth", 3, 7),
                    "n_estimators": 100,
                    "random_state": config['data']['random_state']
                }
                model = xgb.XGBClassifier(**params)
                
            else: # Random Forest
                mlflow.sklearn.autolog(log_models=True)
                params = {
                    "n_estimators": trial.suggest_int("rf_estimators", 50, 200),
                    "max_depth": trial.suggest_int("rf_depth", 5, 15),
                    "random_state": config['data']['random_state']
                }
                model = RandomForestClassifier(**params)
            
            # Fit Model
            if model_type == "xgboost":
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_train, y_train)
                
            # Score
            val_loss = trial.suggest_float("dummy_loss_eval", 0, 1) # Placeholder proxy metric extraction
            # Real evaluation calculations occur in the next phase module
            return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=4)
    
    return study.best_params, X_train, X_val, y_train, y_val
