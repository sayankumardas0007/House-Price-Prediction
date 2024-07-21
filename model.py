import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('house_prices.csv')

# Explore the dataset
print(data.head())

# Define features and target
X = data[['size']]
y = data['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Compare predictions with actual values
print('Predicted prices:', y_pred)
print('Actual prices:', y_test.values)

# Plot the results
plt.scatter(X, y, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('House Price Prediction')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($)')
plt.show()

# Print model coefficients
print('Coefficient:', model.coef_)
print('Intercept:', model.intercept_)
