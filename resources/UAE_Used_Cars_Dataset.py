# -----------------------------------------------
# UAE Used Cars Dataset Analysis
# -----------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# Dataset Overview
# -------------------------------------------------
"""
This dataset contains listings of used cars in the UAE. 
It includes details such as make, model, year, price, mileage, transmission type,
fuel type, engine size, and location. The objective is to analyze the market,
understand pricing patterns, and identify trends affecting used car sales.
"""

# Load dataset
df = pd.read_csv('UAE_Used_Cars_Dataset.csv')

# Show initial info
print("Initial Dataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# -------------------------------------------------
# Data Cleaning
# -------------------------------------------------
"""
We start by handling missing values, checking for duplicates,
and ensuring all columns are of appropriate data types.
"""

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Drop rows with critical missing info (e.g., Price or Make)
df.dropna(subset=['Price', 'Make', 'Model', 'Year'], inplace=True)

# Fill less critical missing values
if 'Mileage' in df.columns:
    df['Mileage'].fillna(df['Mileage'].median(), inplace=True)

# Remove duplicates
duplicates = df.duplicated()
print(f"\nNumber of duplicate rows: {duplicates.sum()}")
df.drop_duplicates(inplace=True)

# Convert Year to integer (if needed)
if df['Year'].dtype != int:
    df['Year'] = df['Year'].astype(int)

# -------------------------------------------------
# Descriptive Statistics
# -------------------------------------------------
"""
Summary statistics provide an understanding of the range,
central tendency, and spread of numeric features.
"""

# Numerical features (customize based on actual columns)
numeric_cols = df.select_dtypes(include='number').columns

print("\nDescriptive Statistics:")
print(df[numeric_cols].describe())

# Mode
print("\nMode Values:")
print(df[numeric_cols].mode().iloc[0])

# Skewness & Kurtosis
print("\nSkewness:")
print(df[numeric_cols].skew())

print("\nKurtosis:")
print(df[numeric_cols].kurt())

# Categorical frequency
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"\nFrequency Table for {col}:")
    print(df[col].value_counts())

# -------------------------------------------------
# Data Visualization
# -------------------------------------------------
"""
We use histograms, boxplots, and correlation plots
to explore data distribution and feature relationships.
"""

# -- Univariate Analysis --

# Histograms
df[numeric_cols].hist(bins=30, figsize=(15, 10))
plt.suptitle("Histograms of Numerical Features")
plt.tight_layout()
plt.show()

# Boxplots by categorical label (e.g., Transmission or Fuel Type if available)
if 'Transmission' in df.columns:
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x='Transmission', y=col, data=df)
        plt.title(f'{col} by Transmission')
        plt.tight_layout()
        plt.show()

# Bar plot for car brands
if 'Make' in df.columns:
    top_makes = df['Make'].value_counts().nlargest(10)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_makes.index, y=top_makes.values)
    plt.title("Top 10 Car Brands by Listings")
    plt.xticks(rotation=45)
    plt.ylabel("Number of Listings")
    plt.tight_layout()
    plt.show()

# -- Bivariate Analysis --

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Pairwise scatter plots
selected_pairs = [('Mileage', 'Price'), ('Year', 'Price')]
for x, y in selected_pairs:
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=x, y=y, data=df)
    plt.title(f'{y} vs. {x}')
    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# Insights and Hypothesis
# -------------------------------------------------
"""
Observations:
1. Newer cars generally have a higher resale value.
2. Mileage negatively correlates with price — cars driven more are cheaper.
3. Certain brands (e.g., Toyota, Nissan) dominate the market.

Hypothesis:
"Car year and mileage are the two most significant predictors of used car price in the UAE."

This could be tested by fitting a regression model or using feature importance in ML classifiers.
"""

input("Press Enter to finish...")
