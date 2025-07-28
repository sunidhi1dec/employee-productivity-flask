import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Employee Productivity Prediction", layout="wide")

st.title("Employee Productivity Prediction")

uploaded_file = st.file_uploader("Upload Garment Worker Productivity CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

    # Drop date column if exists
    if 'date' in df.columns:
        df.drop(columns=['date'], inplace=True)

    # Encode categorical columns
    categorical_cols = ['quarter', 'department', 'day']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Split into features and target
    X = df_imputed.drop(columns=['actual_productivity'])
    y = df_imputed['actual_productivity']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42)
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2})

    results_df = pd.DataFrame(results).round(4)
    st.subheader("Model Comparison Table")
    st.write(results_df)

    # Plot chart
    st.subheader("Performance Comparison Chart")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(results_df['Model']))
    width = 0.25
    ax.bar(x - width, results_df['MAE'], width=width, label='MAE')
    ax.bar(x, results_df['RMSE'], width=width, label='RMSE')
    ax.bar(x + width, results_df['R²'], width=width, label='R²')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'])
    ax.legend()
    st.pyplot(fig)

    # Select model for prediction
    selected_model_name = st.selectbox("Choose model for new predictions", list(models.keys()))
    selected_model = models[selected_model_name]

    st.subheader("Input Features for Prediction")
    input_data = {}
    for col in X.columns:
        val = st.number_input(f"{col}", value=float(df[col].mean()))
        input_data[col] = val

    if st.button("Predict Productivity"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = selected_model.predict(input_scaled)[0]
        st.success(f"Predicted Productivity ({selected_model_name}): {prediction:.4f}")
