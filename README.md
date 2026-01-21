# Feature Encoding & Scaling (Task 4)

## Project Overview
This project focuses on the essential steps of Feature Engineering: cleaning, encoding, and scaling data. Using the "All-in-One" dataset, we prepared raw data for Machine Learning models by converting human-readable categories into mathematical representations and normalizing numerical ranges.

## Datasets Used
**[All-in-one dataset for preprocessing practice](https://www.kaggle.com/datasets/akshatsharma2/all-in-one-dataset-for-preprocessing-practice)** by **Akshat Sharma** (Used in this project).

## Tools & Libraries
- **Python**: Core programming language.
- **Pandas**: For data manipulation and cleaning.
- **Scikit-Learn**: Specifically `StandardScaler` for feature normalization.

## Implementation Steps

### 1. Data Cleaning
- **Identifier Removal**: Dropped the `name` column as it contains no predictive patterns.
- **Missing Value Imputation**: 
  - Numerical (`age`, `cgpa`) filled with **Median** to handle potential outliers.
  - Categorical (`city`, `gender`, `profession`) filled with **Mode** (most frequent value).

### 2. Feature Encoding
- **Ordinal Encoding**: The `profession` column was manually mapped (`bachelor: 0, masters: 1, phd: 2`) to preserve the educational hierarchy.
- **One-Hot Encoding**: The `city` and `gender` columns were transformed using `pd.get_dummies`. 
  - **Dummy Variable Trap**: Applied `drop_first=True` to remove redundant columns and ensure mathematical stability for linear models.

### 3. Feature Scaling
- **Standardization**: Applied `StandardScaler` to `age` and `cgpa`.
- **Result**: Both features now have a **Mean of 0** and a **Standard Deviation of 1**, ensuring the model treats them with equal importance regardless of their original units.

## Final Outcome
The resulting dataset (`processed_data.csv`) is fully optimized for distance-based algorithms (like KNN/SVM) and gradient-descent algorithms (like Logistic Regression).

## How to Run
1. Ensure you have `pandas` and `scikit-learn` installed.
2. Run the Jupyter Notebook `task4.ipynb`.
3. The script will output a cleaned and scaled CSV file ready for training.
