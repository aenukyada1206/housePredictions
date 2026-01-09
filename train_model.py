import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Sample house data
data = {
    "area": [800, 900, 1000, 1100, 1200, 1300, 1400],
    "bedrooms": [1, 2, 2, 3, 3, 3, 4],
    "price": [3000000, 3400000, 3800000, 4200000, 4800000, 5200000, 6000000]
}

df = pd.DataFrame(data)

X = df[["area", "bedrooms"]]
y = df["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ model.pkl created successfully")
