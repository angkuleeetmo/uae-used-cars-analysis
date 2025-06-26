# -------------------------------------------------------------------
# UAE Used Car Price Prediction: An End-to-End Machine Learning Project
#
# This script demonstrates a full workflow:
# 1. Data Loading and Cleaning
# 2. Exploratory Data Analysis (EDA)
# 3. Feature Engineering
# 4. Model Building, Training, and Evaluation
# 5. Feature Importance Analysis
# -------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

# --- 1. Project Setup ---
print("🚀 Starting the UAE Used Car Price Prediction project...")

# Create a directory to save plots if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')
    print("📁 Created 'images' directory for saving plots.")

# --- 2. Data Loading and Cleaning ---
print("\n[STEP 1/5] Loading and Cleaning Data...")
try:
    df = pd.read_csv('UAE_Used_Cars_Dataset.csv')
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'UAE_Used_Cars_Dataset.csv' not found. Please place it in the project root directory.")
    exit()

# Handle missing values
initial_rows = len(df)
df.dropna(subset=['Price', 'Make', 'Model', 'Year', 'Mileage'], inplace=True)
for col in ['Body Type', 'Cylinders', 'Transmission', 'Fuel Type']:
    if col in df.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
print(f"✅ Dropped {initial_rows - len(df)} rows with critical missing data and filled others.")

# Correct data types
df['Year'] = df['Year'].astype(int)
df['Price'] = df['Price'].astype(int)
df['Mileage'] = df['Mileage'].astype(int)
print("✅ Corrected data types for key columns.")

# Remove duplicates
duplicates_found = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"✅ Removed {duplicates_found} duplicate rows. Final dataset has {len(df)} rows.")

# --- 3. Exploratory Data Analysis (EDA) ---
print("\n[STEP 2/5] Performing Exploratory Data Analysis...")

# Set plot style
sns.set_style("whitegrid")
palette = "viridis"

# Top 15 Car Brands
plt.figure(figsize=(14, 8))
top_makes = df['Make'].value_counts().nlargest(15)
sns.barplot(y=top_makes.index, x=top_makes.values, palette=palette, orient='h')
plt.title('Top 15 Car Brands by Number of Listings', fontsize=16)
plt.xlabel('Number of Listings', fontsize=12)
plt.ylabel('Car Make', fontsize=12)
plt.tight_layout()
plt.savefig('images/top_15_brands.png', bbox_inches='tight')
plt.show()

# Correlation Heatmap for numerical features
plt.figure(figsize=(10, 7))
numeric_cols = ['Year', 'Price', 'Mileage']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
plt.savefig('images/correlation_heatmap.png', bbox_inches='tight')
plt.show()

print("✅ EDA plots saved to 'images/' directory.")

# --- 4. Feature Engineering & Model Preparation ---
print("\n[STEP 3/5] Engineering features for the ML model...")

# Drop columns not suitable for this model (Model has too many unique values, Description requires NLP)
df_ml = df.drop(columns=['Model', 'Description', 'Color'])

# One-Hot Encode categorical features
categorical_features = ['Make', 'Body Type', 'Transmission', 'Fuel Type', 'Location', 'Cylinders']
df_ml = pd.get_dummies(df_ml, columns=categorical_features, drop_first=True)
print(f"✅ Data transformed for modeling. Shape of ML-ready data: {df_ml.shape}")

# Define features (X) and target (y)
X = df_ml.drop('Price', axis=1)
y = df_ml['Price']

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows) sets.")

# --- 5. Model Training, Evaluation, and Analysis ---
print("\n[STEP 4/5] Training the Random Forest Regressor model...")
# Initialize and train the model. n_jobs=-1 uses all available CPU cores.
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
rf_model.fit(X_train, y_train)
print("✅ Model training complete.")

print("\n[STEP 5/5] Evaluating model and analyzing features...")
# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Model Performance Metrics ---")
print(f"R-squared (R²): {r2:.3f}")
print(f"Mean Absolute Error (MAE): {mae:,.2f} AED")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f} AED")
print(f"Out-of-Bag (OOB) Score: {rf_model.oob_score_:.3f}")
print("---------------------------------")

# Get and plot feature importances
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(14, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='mako')
plt.title('Top 15 Most Important Features for Price Prediction', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('images/feature_importance.png', bbox_inches='tight')
plt.show()

print("✅ Feature importance plot saved.")
print("\n🎉 Project script finished successfully!")