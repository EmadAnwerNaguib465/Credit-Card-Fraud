# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using various classification algorithms. This project analyzes a dataset of 1,000,000 credit card transactions and implements multiple ML models to identify fraudulent activities with high accuracy.

## 📊 Dataset Overview

The dataset contains **1,000,000 transactions** with the following features:

### Features
- **distance_from_home**: Distance from the cardholder's home location
- **distance_from_last_transaction**: Distance from the previous transaction location
- **ratio_to_median_purchase_price**: Transaction amount compared to median purchase price
- **repeat_retailer**: Binary indicator if the retailer was used before
- **used_chip**: Binary indicator if the chip was used for the transaction
- **used_pin_number**: Binary indicator if PIN was entered
- **online_order**: Binary indicator if the transaction was online

### Target Variable
- **fraud**: Binary indicator (0 = legitimate transaction, 1 = fraudulent transaction)

### Class Distribution
- Fraudulent transactions: **8.74%**
- Legitimate transactions: **91.26%**
- Note: This is an **imbalanced dataset** requiring special handling techniques

## 🔍 Project Workflow

### 1. Data Exploration & Preprocessing
- Loaded and explored the dataset structure
- Checked for missing values (none found)
- Checked for duplicate records (none found)
- Identified outliers in continuous features:
  - Distance from home
  - Distance from last transaction
  - Ratio to median purchase price
- Converted binary features to integer type for better performance
- Applied outlier treatment using IQR (Interquartile Range) method

### 2. Feature Engineering
- Handled outliers using capping/flooring techniques
- Applied feature scaling using StandardScaler
- Prepared features for model training

### 3. Model Training & Evaluation

Multiple machine learning algorithms were implemented and evaluated:

#### Models Tested:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **XGBoost Classifier**

#### Evaluation Metrics:
- Accuracy Score
- F1 Score (important for imbalanced datasets)
- Classification Report (Precision, Recall, F1-score for each class)
- Confusion Matrix
- Training vs Testing performance comparison

### 4. Model Performance Visualization
- Confusion matrix heatmaps for each model
- Performance comparison across models
- Training vs testing accuracy analysis

## 🛠️ Technologies & Libraries Used

### Core Libraries
```python
- pandas - Data manipulation and analysis
- numpy - Numerical computations
- matplotlib - Data visualization
- seaborn - Statistical data visualization
- plotly - Interactive visualizations
```

### Machine Learning
```python
- scikit-learn - ML algorithms and utilities
  - Classification models (Logistic Regression, Decision Tree, Random Forest)
  - Preprocessing tools (StandardScaler, MinMaxScaler)
  - Model evaluation metrics
  - Train-test splitting and cross-validation
- xgboost - Gradient boosting classifier
```

### Model Persistence
```python
- joblib - Saving and loading trained models
```

## 📁 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── Credit_Card_Fraud.ipynb    # Main Jupyter notebook
├── card_transdata.csv         # Dataset (1M transactions)
├── README.md                  # Project documentation
└── models/                    # Saved trained models (if applicable)
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.x
Jupyter Notebook or JupyterLab
```

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd Credit-Card-Fraud-Detection
```

2. Install required packages
```bash
pip install numpy pandas matplotlib seaborn plotly scikit-learn xgboost joblib
```

3. Launch Jupyter Notebook
```bash
jupyter notebook Credit_Card_Fraud.ipynb
```

## 📈 Key Findings

- The dataset exhibits **class imbalance** with only 8.74% fraudulent transactions
- Multiple features show **significant outliers** requiring careful preprocessing
- Feature scaling is essential for optimal model performance
- Ensemble methods (Random Forest, XGBoost) typically perform better on imbalanced datasets
- F1-score is a more reliable metric than accuracy for this imbalanced classification problem

## 🎯 Model Evaluation Strategy

A custom `pred_report()` function was created to evaluate each model:
- Predictions on both training and testing sets
- Accuracy and F1-score calculation
- Detailed classification report
- Confusion matrix visualization
- Performance comparison between training and testing data

## 🔮 Future Improvements

- [ ] Implement SMOTE or other resampling techniques for class imbalance
- [ ] Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
- [ ] Feature importance analysis to identify key fraud indicators
- [ ] Implement additional algorithms (SVM, Neural Networks)
- [ ] Deploy the best model as a web application or API
- [ ] Add real-time fraud detection capabilities
- [ ] Implement model monitoring and drift detection

## 📝 License

This project is available for educational and research purposes.

## 👥 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This project is designed for educational purposes to demonstrate machine learning techniques for fraud detection. Always ensure compliance with data privacy regulations when working with financial data.
