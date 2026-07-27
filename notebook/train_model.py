"""
Laptop Price Prediction - Model Training
=========================================
Follows the exact pipeline described in the project documentation:
Data Collection -> Library Import -> Preprocessing -> Encoding ->
Train-Test Split -> Model Training -> Model Evaluation -> Save Best Model
"""

# -----------------------------
# 1. Importing Libraries
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# 2. Reading the Dataset
# -----------------------------
data = pd.read_csv(os.path.join(BASE_DIR, "data", "laptop_price.csv"), encoding="windows-1252")
print("Dataset shape:", data.shape)

# -----------------------------
# 3. Dropping Unnecessary Columns
# -----------------------------
data.drop(['laptop_ID'], axis=1, inplace=True)

# -----------------------------
# 4. Displaying Data for Inspection
# -----------------------------
print(data.head())
print(data.tail())

# -----------------------------
# 5. Data Overview
# -----------------------------
print(data.info())

# -----------------------------
# 6. Null Value Detection
# -----------------------------
print(data.isnull().sum())

# -----------------------------
# 7 & 8. Checking Unique Values / Frequency Counts
# -----------------------------
print(data['TypeName'].unique())
a = data['TypeName'].value_counts()

plt.figure()
plt.pie(a, labels=a.index, autopct='%1.1f%%', startangle=90,
        colors=['y', 'r', 'b', 'g', 'c', 'm'])
plt.title("Distribution of Laptop Types")
plt.savefig(os.path.join(BASE_DIR, "notebook", "typename_distribution.png"))
plt.close()

print(data['Product'].unique())
print(data['OpSys'].unique())
print(data['Gpu'].unique())

b = data['Product'].value_counts()
c = data['OpSys'].value_counts()
d = data['Gpu'].value_counts()

plt.figure()
plt.bar(c.index, c, color=['r', 'g', 'y', 'b', 'c', 'm', 'k', 'w', 'r'])
plt.title("Operating System Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "notebook", "opsys_distribution.png"))
plt.close()

# -----------------------------
# 9. Label Encoding (Categorical Encoding)
# -----------------------------
# We save each column's encoder separately so the Flask app can
# apply the SAME encoding to new user input at prediction time.
categorical_cols = ['TypeName', 'Company', 'Ram', 'Gpu', 'ScreenResolution',
                     'Cpu', 'Memory', 'Product', 'OpSys', 'Weight']

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# -----------------------------
# 10. Train-Test Split
# -----------------------------
X = data.drop('Price_euros', axis=1)
y = data['Price_euros']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# -----------------------------
# 11. Model Training
# -----------------------------
# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)

# Model 2: Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, max_depth=14, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# -----------------------------
# 12. Model Evaluation (R^2, MAE)
# -----------------------------
lr_r2 = r2_score(y_test, lr_preds)
lr_mae = mean_absolute_error(y_test, lr_preds)

rf_r2 = r2_score(y_test, rf_preds)
rf_mae = mean_absolute_error(y_test, rf_preds)

print("\n----- Model Evaluation -----")
print(f"Linear Regression -> R2: {lr_r2:.4f} | MAE: {lr_mae:.2f}")
print(f"Random Forest      -> R2: {rf_r2:.4f} | MAE: {rf_mae:.2f}")

# -----------------------------
# 13. Best Model Selection
# -----------------------------
if rf_r2 >= lr_r2:
    best_model = rf_model
    best_name = "Random Forest Regression"
else:
    best_model = lr_model
    best_name = "Linear Regression"

print(f"\nBest model selected: {best_name}")

# -----------------------------
# 14. Saving the Model + Encoders
# -----------------------------
with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as f:
    pickle.dump(best_model, f)

with open(os.path.join(BASE_DIR, "encoders.pkl"), "wb") as f:
    pickle.dump(encoders, f)

with open(os.path.join(BASE_DIR, "feature_columns.pkl"), "wb") as f:
    pickle.dump(list(X.columns), f)

print("\nSaved model.pkl, encoders.pkl, feature_columns.pkl")
print("Training complete.")
