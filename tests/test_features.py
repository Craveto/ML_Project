import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import preprocess_telco_data

def test_preprocess_telco_data(tmp_path):
    # 1. Create a tiny mock version of your IBM Telco dataset
    mock_data = pd.DataFrame({
        "customerID": ["1234-ABCD", "5678-EFGH"],
        "gender": ["Female", "Male"],
        "TotalCharges": ["123.4", " "],  # Includes the empty space quirk
        "Churn": ["Yes", "No"]
    })
    
    # Save it to a temporary path provided by pytest
    test_file = tmp_path / "mock_telco.csv"
    mock_data.to_csv(test_file, index=False)
    
    # 2. Run your feature engineering module logic
    X, y = preprocess_telco_data(str(test_file), target_col="Churn", id_col="customerID")
    
    # 3. Assertions to verify the engineering logic worked perfectly
    assert "customerID" not in X.columns, "❌ customerID column was not dropped!"
    assert y.iloc[0] == 1, "❌ Churn 'Yes' was not mapped to 1!"
    assert y.iloc[1] == 0, "❌ Churn 'No' was not mapped to 0!"
    assert not X["TotalCharges"].isnull().values.any(), "❌ Empty space in TotalCharges was not filled!"
    assert X["TotalCharges"].dtype == np.float64, "❌ TotalCharges was not converted to a float numeric type!"
    print("✅ All feature engineering unit tests passed successfully!")
