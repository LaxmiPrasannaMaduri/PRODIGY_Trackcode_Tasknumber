import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
data = {
    'SquareFootage': [1200, 1500, 1700, 2000, 2200, 2500, 2800, 3000, 3500, 4000],
    'Bedrooms': [2, 3, 3, 4, 4, 4, 5, 5, 5, 6],
    'Bathrooms': [1, 2, 2, 2, 3, 3, 3, 4, 4, 5],
    'Price': [150000, 200000, 210000, 250000, 275000, 300000, 350000, 400000, 450000, 500000]
}

df = pd.DataFrame(data)   
X = df[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predicted Prices:", y_pred)
print("Actual Prices:", list(y_test))
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R² Score:", r2)

new_house = np.array([[2600, 4, 3]])
predicted_price = model.predict(new_house)
print("Predicted Price for new house:", predicted_price[0])
