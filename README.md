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


