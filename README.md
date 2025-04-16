# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load data, select features/targets, and split into train/test sets.

2. Scale features and targets using StandardScaler.

3. Train SGDRegressor with MultiOutputRegressor on training data.

4. Predict, inverse scale, and compute MSE.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.


import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = fetch_california_housing()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df['HousingPrice'] = dataset.target
print("Sample Data:\n", df.head())

# Feature and Target Selection
X = df.drop(columns=['AveOccup', 'HousingPrice'])  # Features
Y = df[['AveOccup', 'HousingPrice']]               # Multi-output targets

# Data Splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature Scaling
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

# Train Multivariate Linear Regression using SGD
sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train, Y_train)

# Make Predictions and Inverse Transform
Y_pred = multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

# Evaluation
mse = mean_squared_error(Y_test, Y_pred)
print("\nMean Squared Error:", mse)
print("\nSample Predictions:\n", Y_pred[:5])
```

## Output:

## California Housing Dataset
![Screenshot 2025-04-16 213242](https://github.com/user-attachments/assets/666d3138-4785-439b-8ca7-2c30ae25475d)

## Features and Target
![Screenshot 2025-04-16 213259](https://github.com/user-attachments/assets/4208ccf8-9cff-4653-a10c-0d8b318f1e5b)

## Predictions and Evaluate
![Screenshot 2025-04-16 213308](https://github.com/user-attachments/assets/c8fd8dbc-a13a-4414-802e-6f91b0b8e437)

## Developed by : BALA SARAVANAN K
## Reg no: 24900611


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
