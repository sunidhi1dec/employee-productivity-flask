from flask import Flask, render_template, request, jsonify
import os
import pickle
from model_utils import train_models
import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__, template_folder='templates')

# Allow parent folder imports if run from Flask folder
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/app')
def app_page():
    return render_template('app.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    # Train models
    results, features = train_models(filepath)

    df_metrics = pd.DataFrame(results).T

    # --- Bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    df_metrics.plot(kind='bar', ax=ax)
    plt.title('Model Comparison (MAE, RMSE, R²)')
    plt.xticks(rotation=20)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    bar_chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(5, 4))
    corr = df_metrics.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Metrics Correlation Heatmap')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    heatmap_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return jsonify({
        "results": results,
        "features": features,
        "bar_chart": bar_chart_base64,
        "heatmap": heatmap_base64
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_name = data['model']
    features = data['features']

    model_path = os.path.join("IBM Files", f"{model_name.replace(' ', '_')}.pkl")
    scaler_path = os.path.join("IBM Files", "scaler.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    df = pd.DataFrame([features])
    scaled = scaler.transform(df)
    pred = model.predict(scaled)[0]
    return jsonify({"prediction": float(pred)})

if __name__ == '__main__':
    app.run(debug=True)
