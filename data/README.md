# 📊 Datasets - Classical Machine Learning

> **"Data is the new oil, but it's only valuable when refined."** - Inspired by Terence Tao's approach to data analysis

## 🌟 Overview

This directory contains carefully curated datasets for hands-on learning and practice. Each dataset is designed to illustrate specific machine learning concepts and provide realistic challenges.

## 📚 Dataset Categories

### **Phase 1: Mathematical Foundations**
- [Linear Algebra Datasets](#phase-1-linear-algebra-datasets)
- [Probability and Statistics Datasets](#phase-1-probability-and-statistics-datasets)
- [Optimization Datasets](#phase-1-optimization-datasets)

### **Phase 2: Supervised Learning**
- [Regression Datasets](#phase-2-regression-datasets)
- [Classification Datasets](#phase-2-classification-datasets)
- [Time Series Datasets](#phase-2-time-series-datasets)

### **Phase 3: Unsupervised Learning**
- [Clustering Datasets](#phase-3-clustering-datasets)
- [Dimensionality Reduction Datasets](#phase-3-dimensionality-reduction-datasets)
- [Association Rules Datasets](#phase-3-association-rules-datasets)

### **Phase 4: Reinforcement Learning**
- [Game Environments](#phase-4-game-environments)
- [Control Problems](#phase-4-control-problems)
- [Trading Environments](#phase-4-trading-environments)

### **Phase 5: Predictive Analytics**
- [Business Datasets](#phase-5-business-datasets)
- [Real-World Applications](#phase-5-real-world-applications)
- [Synthetic Datasets](#phase-5-synthetic-datasets)

## 📊 Dataset Information

### **Dataset Format**
- **CSV**: Comma-separated values for tabular data
- **JSON**: JavaScript Object Notation for structured data
- **Parquet**: Columnar format for efficient storage
- **HDF5**: Hierarchical Data Format for large datasets

### **Data Quality**
- **Clean**: No missing values or errors
- **Realistic**: Based on real-world scenarios
- **Balanced**: Appropriate class distributions
- **Documented**: Clear descriptions and metadata

### **Privacy and Ethics**
- **Anonymized**: No personal identifying information
- **Synthetic**: Generated data when privacy is a concern
- **Public**: Openly available datasets
- **Ethical**: Collected and used responsibly

## 🎯 Phase 1: Mathematical Foundations

### **Linear Algebra Datasets**
- **`matrix_operations.csv`**: Matrices for linear algebra exercises
- **`eigenvalue_problems.csv`**: Matrices for eigenvalue decomposition
- **`pca_examples.csv`**: High-dimensional data for PCA practice

### **Probability and Statistics Datasets**
- **`normal_distributions.csv`**: Samples from normal distributions
- **`binomial_experiments.csv`**: Binomial distribution examples
- **`correlation_data.csv`**: Correlated variables for analysis

### **Optimization Datasets**
- **`convex_functions.csv`**: Data for convex optimization
- **`gradient_descent.csv`**: Examples for gradient descent
- **`constraint_optimization.csv`**: Constrained optimization problems

## 🎯 Phase 2: Supervised Learning

### **Regression Datasets**
- **`house_prices.csv`**: Real estate prices with features
- **`stock_prices.csv`**: Historical stock price data
- **`weather_data.csv`**: Temperature and weather predictions
- **`sales_forecasting.csv`**: Retail sales data for forecasting

### **Classification Datasets**
- **`iris.csv`**: Classic iris flower classification
- **`wine_quality.csv`**: Wine quality assessment
- **`customer_churn.csv`**: Customer retention prediction
- **`spam_detection.csv`**: Email spam classification
- **`medical_diagnosis.csv`**: Disease diagnosis prediction

### **Time Series Datasets**
- **`stock_market.csv`**: Daily stock prices
- **`energy_consumption.csv`**: Hourly energy usage
- **`website_traffic.csv`**: Daily website visitors
- **`temperature_records.csv`**: Daily temperature measurements

## 🎯 Phase 3: Unsupervised Learning

### **Clustering Datasets**
- **`customer_segmentation.csv`**: Customer behavior data
- **`gene_expression.csv`**: Gene expression patterns
- **`image_pixels.csv`**: Image pixel data for clustering
- **`anomaly_detection.csv`**: Data with outliers and anomalies

### **Dimensionality Reduction Datasets**
- **`high_dimensional.csv`**: High-dimensional feature data
- **`text_embeddings.csv`**: Word embeddings for NLP
- **`image_features.csv`**: Image feature vectors
- **`genomic_data.csv`**: Genomic sequence data

### **Association Rules Datasets**
- **`market_basket.csv`**: Shopping basket transactions
- **`movie_ratings.csv`**: User movie rating data
- **`product_recommendations.csv`**: E-commerce transaction data
- **`web_navigation.csv`**: Website navigation patterns

## 🎯 Phase 4: Reinforcement Learning

### **Game Environments**
- **`tictactoe.csv`**: Tic-tac-toe game states
- **`chess_positions.csv`**: Chess board positions
- **`poker_hands.csv`**: Poker hand data
- **`maze_navigation.csv`**: Maze solving problems

### **Control Problems**
- **`cart_pole.csv`**: Cart-pole balancing data
- **`mountain_car.csv`**: Mountain car control data
- **`pendulum.csv`**: Pendulum swing-up data
- **`quadcopter.csv`**: Quadcopter control data

### **Trading Environments**
- **`trading_signals.csv`**: Financial trading signals
- **`portfolio_optimization.csv`**: Portfolio allocation data
- **`risk_management.csv`**: Risk assessment data
- **`market_making.csv`**: Market making strategies

## 🎯 Phase 5: Predictive Analytics

### **Business Datasets**
- **`customer_lifetime_value.csv`**: Customer value prediction
- **`supply_chain.csv`**: Supply chain optimization
- **`marketing_campaigns.csv`**: Marketing effectiveness
- **`operational_efficiency.csv`**: Operations optimization

### **Real-World Applications**
- **`healthcare_outcomes.csv`**: Medical outcome prediction
- **`energy_optimization.csv`**: Energy consumption optimization
- **`traffic_prediction.csv`**: Traffic flow prediction
- **`fraud_detection.csv`**: Financial fraud detection

### **Synthetic Datasets**
- **`simulated_sales.csv`**: Synthetic sales data
- **`generated_users.csv`**: Synthetic user behavior
- **`mock_transactions.csv`**: Synthetic transaction data
- **`artificial_signals.csv`**: Synthetic signal data

## 🚀 Getting Started

### **1. Download Datasets**
```bash
# Download all datasets
python download_datasets.py

# Download specific phase
python download_datasets.py --phase 1

# Download specific dataset
python download_datasets.py --dataset house_prices
```

### **2. Load Datasets**
```python
import pandas as pd

# Load a dataset
df = pd.read_csv('data/phase2/house_prices.csv')

# Explore the data
print(df.head())
print(df.info())
print(df.describe())
```

### **3. Data Preprocessing**
```python
# Handle missing values
df = df.dropna()

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])

# Scale numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])
```

## 📝 Dataset Documentation

Each dataset includes:
- **Description**: What the data represents
- **Source**: Where the data came from
- **Features**: Description of each column
- **Target**: What we're trying to predict
- **Size**: Number of rows and columns
- **Quality**: Data quality assessment
- **Usage**: Suggested exercises and applications

## 🔒 Privacy and Ethics

- **No Personal Data**: All datasets are anonymized
- **Public Sources**: Data from publicly available sources
- **Synthetic Data**: Generated data when privacy is a concern
- **Ethical Use**: Data used responsibly and ethically
- **Attribution**: Proper attribution to data sources

## 📞 Support and Resources

- **Data Dictionary**: Detailed description of each dataset
- **Tutorials**: Step-by-step data analysis guides
- **Examples**: Sample code for each dataset
- **Community**: Discussion forums for data questions
- **Updates**: Regular updates and new datasets

---

**Ready to start working with data? Begin with the [Phase 1 Datasets](phase1/)!** 🚀
