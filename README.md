# Employee Productivity Prediction - Flask Web Application

This project is a **Flask-based web application** that predicts employee productivity using machine learning models.
It allows users to upload a dataset, preprocess data, compare models (Linear Regression, Random Forest, XGBoost),
visualize results, and predict productivity for new inputs through a user-friendly interface.

---

## **Features**

- Upload CSV dataset
- Automatic preprocessing of data
- Train and evaluate models:
  - Linear Regression
  - Random Forest
  - XGBoost
- Comparative analysis with metrics (MAE, RMSE, R²)
- Visualization:
  - Bar chart of model performance
  - Heatmap of correlations
- Predict productivity for new inputs

---

## **Project Structure**

Flask/
├── static/
│ ├── css/
│ └── images/
├── templates/
│ ├── index.html
│ └── app.html
├── training/
│ ├── model_utils.py
│ └── saved_models/
├── uploads/
│ └── preprocessed_garments_productivity.csv
├── app.py
├── requirements.txt
└── README.md

yaml
Copy
Edit


---

## **Tech Stack**

- **Frontend:** HTML, CSS, Bootstrap
- **Backend:** Python (Flask)
- **ML Libraries:** pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

---

## **Setup Instructions**

### 1. Clone the repository

```bash
git clone https://github.com/sunidhi1dec/employee-productivity-flask.git
cd employee-productivity-flask

2. Create and activate a virtual environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
venv\Scripts\activate   # On Windows
# source venv/bin/activate  # On macOS/Linux

3. Install dependencies

pip install -r requirements.txt

4. Run the Flask application

bash
Copy
Edit
python app.py

Usage
Open the app in your browser at http://127.0.0.1:5000/.

Upload your dataset or use the pre-loaded one.

View data preprocessing results.

Train models and compare performance metrics.

Use the prediction form to predict employee productivity.

Dataset
The application is designed for the Garments Worker Productivity dataset but can be adapted for similar datasets.
Features include:

date, quarter, department, day, team

targeted_productivity, smv, wip, over_time, incentive

idle_time, idle_men, no_of_style_change, actual_productivity

Future Enhancements
Deploy the app to a cloud platform (Heroku/AWS)

Add user authentication

Real-time data integration

Enhanced dashboards
