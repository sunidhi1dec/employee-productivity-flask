import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

def train_models(csv_file):
    # Load dataset
    df = pd.read_csv(csv_file)

    # Drop 'date' column if present
    if 'date' in df.columns:
        df.drop(columns=['date'], inplace=True)

    # Encode categorical columns
    categorical_cols = ['quarter', 'department', 'day']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Split features and target
    X = df_imputed.drop(columns=['actual_productivity'])
    y = df_imputed['actual_productivity']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42)
    }

    results = {}

    # Save models in IBM Files
    os.makedirs("IBM Files", exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

        model_filename = os.path.join("IBM Files", f"{name.replace(' ', '_')}.pkl")
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)

    # Save scaler
    with open(os.path.join("IBM Files", "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    return results, list(X.columns)
