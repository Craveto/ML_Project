import mlflow.pyfunc
import pandas as pd
import numpy as np

class TelcoChurnPyFuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, trained_model, training_columns):
        """
        trained_model: The final fitted model (XGBoost/RF)
        training_columns: List of columns the model saw after One-Hot Encoding
        """
        self.model = trained_model
        self.training_columns = training_columns

    def load_context(self, context):
        """Used if loading external artifacts (not needed for this step)"""
        pass

    def _preprocess_raw_input(self, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Replicates the exact feature engineering steps from Phase 1 & 2
        on raw incoming production JSON/DataFrame requests.
        """
        df = model_input.copy()
        
        # 1. Drop identifier if present in production request
        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])
            
        # 2. Fix TotalCharges formatting string spaces
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
            # In production, handle missing values (fallback to 0 or hardcoded median)
            df["TotalCharges"] = df["TotalCharges"].fillna(0)
            
        # 3. One-Hot Encode categorical variations
        df_encoded = pd.get_dummies(df, drop_first=True)
        
        # 4. CRITICAL MLOPS STEP: Align production columns with training columns
        # Add missing columns that were present in training but missing in this specific production payload
        for col in self.training_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
                
        # Reorder columns to match the exact matrix structure expected by the model
        df_encoded = df_encoded[self.training_columns]
        
        return df_encoded

    def predict(self, context, model_input):
        """
        Invoked automatically when calling model.predict() on the deployed endpoint.
        """
        # Run preprocessing on raw incoming data
        processed_df = self._preprocess_raw_input(model_input)
        
        # Return final predictions (0 or 1)
        return self.model.predict(processed_df)
