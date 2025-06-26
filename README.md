# 🚗 UAE Used Car Price Prediction  - Jovs / CS Elective 4

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=for-the-badge&logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-yellow?style=for-the-badge&logo=pandas)

Made for my Data Science Portfolio Project requirement.

A machine learning project to predict the price of used cars in the United Arab Emirates. This project was developed as a data science portfolio piece to demonstrate a complete workflow from data exploration to model evaluation and interpretation.


## 📖 Table of Contents

- [🎯 About The Project](#-about-the-project)
- [🛠️ Tech Stack](#️-tech-stack)
- [📊 Key Visualizations](#-key-visualizations)
- [🚀 Getting Started](#-getting-started)
- [🤖 Model Performance](#-model-performance)
- [🏁 Conclusion](#-conclusion)
- [📝 License](#-license)

## 🎯 About The Project

The goal of this project is to build a reliable machine learning model that can predict the resale value of a used car in the UAE based on its specifications. By analyzing a dataset of over 2,500 cleaned vehicle listings, this project provides actionable insights for both buyers and sellers in the dynamic UAE automotive market.

**Key objectives include:**
-   Cleaning and preprocessing a real-world dataset of used car listings.
-   Conducting Exploratory Data Analysis (EDA) to uncover market trends and feature relationships.
-   Engineering categorical and numerical features for a regression model.
-   Training and evaluating a Random Forest Regressor to predict car prices.
-   Analyzing feature importances to understand what drives car valuation in the UAE.

## 🛠️ Tech Stack

This project is built using the following core Python libraries for data science:
*   **Pandas:** For data manipulation, cleaning, and preparation.
*   **Matplotlib & Seaborn:** For creating high-quality data visualizations.
*   **Scikit-learn:** For building and evaluating the machine learning model.

## 📊 Key Visualizations

#### Top 15 Most Listed Car Brands
The EDA clearly shows that Japanese and German brands like Toyota, Nissan, and Mercedes-Benz dominate the used car market.
![Top Brands](images/top_15_brands.png)

#### Feature Correlation Heatmap
The heatmap highlights the strong negative correlation between `Price` and `Mileage` (-0.46) and the strong positive correlation between `Price` and `Year` (0.33).


## 🚀 Getting Started

Follow these simple steps to set up and run this project on your local machine.

### Prerequisites
Ensure you have **Python 3.8+** and `pip` installed on your system.

### Installation & Usage

1.  **Clone the Repository**
    ```sh
    git clone https://github.com/your_username/uae-used-cars-analysis.git
    cd uae-used-cars-analysis
    ```
    *(Remember to replace `your_username/uae-used-cars-analysis` with your actual GitHub repository link.)*

2.  **Create a Virtual Environment** (Recommended)
    *   On Windows:
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Required Libraries**
    ```sh
    pip install pandas matplotlib seaborn scikit-learn
    ```

4.  **Place the Dataset**
    <br>Download the `UAE_Used_Cars_Dataset.csv` file and ensure it is in the root directory of the project.

5.  **Run the Main Script**
    ```sh
    python price_prediction_model.py
    ```
    The script will execute the entire pipeline, print key information to the console, and save all generated plots to the `images/` folder.

## 🤖 Model Performance

A **Random Forest Regressor** was trained to predict car prices. Its performance on unseen test data was evaluated using standard regression metrics.

| Metric                      | Score              |
| --------------------------- | ------------------ |
| **R-squared (R²)**          | **0.932**          |
| Out-of-Bag (OOB) Score      | 0.929              |
| **Mean Absolute Error (MAE)** | **40,022.61 AED**  |

An **R-squared score of 0.932** indicates that the model successfully explains over 93% of the variance in car prices, demonstrating a very strong predictive capability. The MAE shows that, on average, the model's price prediction is off by approximately 40,000 AED.

## 🏁 Conclusion

This project successfully developed an accurate model for predicting used car prices in the UAE.

-   **Key Predictors:** The model confirmed that a car's **Year** of manufacture and its **Mileage** are by far the most influential factors in determining its resale value.
-   **Brand Impact:** High-end brands like `porsche`, `land-rover`, and `mercedes-benz` also contribute significantly to higher price predictions.
-   **Practical Application:** This model serves as a powerful tool for buyers to verify fair pricing and for sellers to position their vehicles competitively in the market.

Future work could enhance the model by incorporating NLP on the `Description` text to extract condition details (e.g., 'accident history') and by tuning the model's hyperparameters for even greater accuracy.

## 📝 License

Distributed under the MIT License. See `LICENSE.txt` for more information.
