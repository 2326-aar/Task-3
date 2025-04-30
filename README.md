# Task-3
# 🏠 Housing Price Prediction using Linear Regression

This project implements **Simple** and **Multiple Linear Regression** models to predict housing prices. The task is part of an AI & ML internship, focusing on regression modeling, evaluation metrics, and model interpretation. The dataset used is a housing price dataset, and the process follows the standard procedure for linear regression tasks.

---

## 📁 Dataset

The dataset `Housing.csv` contains various housing features, such as:

- `price`, `area`, `bedrooms`, `bathrooms`
- Categorical features like: `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`

The dataset can be downloaded from the link provided.

---

## 📌 Tasks Covered

### 1. **Data Preprocessing**
   - Load the dataset
   - Handle categorical variables using one-hot encoding
   - Remove outliers
   - Log-transform skewed features

### 2. **Data Splitting**
   - Split data into training and testing sets (80% training, 20% testing)

### 3. **Model Training**
   - Fit a **Linear Regression** model using `sklearn.linear_model`
   - Train both simple and multiple regression models

### 4. **Model Evaluation**
   - Evaluate the model using:
     - **Mean Absolute Error (MAE)**
     - **Mean Squared Error (MSE)**
     - **R² (R-squared)**

### 5. **Visualization**
   - Plot regression lines and compare actual vs predicted values.

---

## 🚀 How to Run

1. Place the `Housing.csv` file in the same folder as the script.
2. Run the Python file:

```bash
python housing_price_prediction.py
