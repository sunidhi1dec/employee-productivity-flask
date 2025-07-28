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

# Load dataset
df = pd.read_csv("preprocessed_garments_productivity.csv")

# Drop date column
df.drop(columns=['date'], inplace=True)

# Encode categorical columns
categorical_cols = ['quarter', 'department', 'day']
df[categorical_cols] = df[categorical_cols].apply(LabelEncoder().fit_transform)

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

# Train and evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R²": r2})

results_df = pd.DataFrame(results).round(4)
print("Model Comparison:\n", results_df)

# Visualization
x = np.arange(len(results_df['Model']))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, results_df['MAE'], width=width, label='MAE')
plt.bar(x, results_df['RMSE'], width=width, label='RMSE')
plt.bar(x + width, results_df['R²'], width=width, label='R²')

plt.xticks(x, results_df['Model'])
plt.title("Model Performance Comparison")
plt.ylabel("Metric Values")
plt.legend()
plt.tight_layout()
plt.show()
