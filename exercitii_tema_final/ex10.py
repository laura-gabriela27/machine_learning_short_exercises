"""10. Utilizați setul de date wine și aplicați DBSCAN pentru a grupa datele,
apoi aplicați regresia liniară pentru a prezice variabila țintă 'target'."""

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

wine = load_wine() # Load the wine dataset
data = pd.DataFrame(data=wine.data, columns=wine.feature_names) # Create a dataframe from the wine dataset
data['target'] = wine.target # Add the target column to the dataframe

X = data.drop('target', axis=1) # Extract the features
y = data['target'] # Extract the target

scaler = StandardScaler() # Initialize the StandardScaler
X_scaled = scaler.fit_transform(X) # Scale the features

dbscan = DBSCAN(eps=2, min_samples=5) # Initialize the DBSCAN model
clusters = dbscan.fit_predict(X_scaled) # Fit the model and predict the clusters

data['cluster'] = clusters # Add the cluster column to the dataframe

print(data) # Print the dataframe
print(data['cluster'].unique()) # Print the unique clusters

# Train the Linear Regression
X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), y, test_size=0.2, random_state=42)
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

y_pred = linear_regression.predict(X_test) # Make predictions

# Evaluate the performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)