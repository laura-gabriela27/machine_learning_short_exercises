"""12. Utilizați setul de date breast cancer și aplicați K-Means pentru a grupa
datele, apoi aplicați regresia liniară pentru a prezice variabila țintă 'target',
iar la final aplicați PCA."""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

breast_cancer = load_breast_cancer() # Load the breast cancer dataset
data = pd.DataFrame(data=breast_cancer.data, columns=breast_cancer.feature_names) # Create a dataframe from the breast cancer dataset
data['target'] = breast_cancer.target # Add the target column to the dataframe

X = data.drop('target', axis=1) # Extract the features
y = data['target'] # Extract the target

scaler = StandardScaler() # Initialize the StandardScaler
X_scaled = scaler.fit_transform(X) # Scale the features

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10) # Initialize the K-Means model
clusters = kmeans.fit_predict(X_scaled) # Fit the model and predict the clusters
data['cluster'] = clusters # Add the cluster column to the dataframe

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # Split the data into training and testing sets

# Train the Linear Regression model
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

y_pred = linear_regression.predict(X_test) # Make predictions

# Evaluate the performance of the Linear Regression model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

# Apply PCA
pca = PCA(n_components=2) # Initialize the PCA model
X_pca = pca.fit_transform(X_scaled) # Fit the model and transform the data

result_df = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Cluster': clusters, 'Target': y}) # Create a DataFrame with the reduced data

# Visualize the reduced data
plt.figure(figsize=(12, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=result_df, palette='viridis')
plt.title('Principal Component Analysis (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
