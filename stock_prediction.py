import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("dataset.csv")

# Convert date column to numbers
data['Day'] = np.arange(len(data))

X = data[['Day']]
y = data['Price']

# Train ML model
model = LinearRegression()
model.fit(X, y)

# Predict next 5 days
future_days = np.array([[len(data) + i] for i in range(5)])
predictions = model.predict(future_days)

# Print predictions
print("Predicted Prices for Next 5 Days:")
for i, price in enumerate(predictions):
    print(f"Day {i+1}: {price:.2f}")

# Plot results
plt.plot(data['Day'], y, label="Actual Prices")
plt.plot(future_days, predictions, label="Predicted Prices", linestyle="dashed")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction")
plt.legend()
plt.show()

