"""7. Folosiți regresia liniară pentru a prezice prețurile caselor în setul de date
California Housing"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Function to calculate Root Mean Squared Error (RMSE)
def get_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Load the California housing dataset
california_dataset = fetch_california_housing()

# Create a DataFrame from the dataset
df = pd.DataFrame(california_dataset.data, columns=california_dataset.feature_names)
df['MEDV'] = california_dataset.target  # Add the target variable 'MEDV' (median house value)

# Split the dataset into training and testing sets
X_train, X_predict, y_train, y_predict = train_test_split(california_dataset.data, california_dataset.target, test_size=(100-80)/100, random_state=42)

# Initialize and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
model_output = model.predict(X_predict)

# Calculate the Root Mean Squared Error (RMSE) of the predictions
pe = get_rmse(predictions=model_output, targets=y_predict)
print(f"Prediction error (RMSE): {pe}")

# Plotting the results
plt.subplot(211)
# Show prediction over test data
t = range(1, len(model_output) + 1)
plt.plot(t, y_predict, 'pink')  # Plot actual values
plt.plot(t, model_output, 'purple')  # Plot predicted values
plt.legend(['Target', 'Prediction'])
plt.ylabel("Housing prices")
plt.title("California Dataset median housing value")

plt.subplot(212)
# Plot prediction error
prediction_error = np.sqrt(np.power(model_output - y_predict, 2))
plt.plot(prediction_error, 'pink')
plt.legend([f"RMSE: {pe}"])
plt.xlabel("x (samples)")
plt.ylabel("Prediction error (RMSE)")
plt.show()
