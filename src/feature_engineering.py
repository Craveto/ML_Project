import pandas as pd
import numpy as np
import os 

def preprocess_telco_data(file_path: str, target_col: str, id_col: str) -> tuple:
    print("⚙️ Executing feature engineering pipeline...")
    df = pd.read_csv(file_path)
    
    # 1. Drop identifiers
    if id_col in df.columns:
        df = df.drop(columns=[id_col])
        
    # 2. Fix TotalCharges formatting space strings to median floats
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    
    # 3. Label encode binary text target column 
    df[target_col] = df[target_col].map({"Yes": 1, "No": 0})
    
    # 4. Generate one-hot encoded variables for categorical attributes
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Clean up local raw file artifact
    if os.path.exists(file_path):
        os.remove(file_path)
        
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    
    return X, y
