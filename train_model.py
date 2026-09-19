import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score


# 1. Load dataset
data = pd.read_csv("dataset/sales.csv")


# 2. Select input and target
X = data[["Quantity", "UnitPrice"]]
y = data["TotalRevenue"]


# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 4. Scale the input features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 5. Create SVR model
model = SVR(kernel="rbf")


# 6. Train the model
model.fit(X_train, y_train)


# 7. Make predictions
y_pred = model.predict(X_test)


# 8. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation")
print("----------------")
print("MAE:", mae)
print("R2 Score:", r2)


# 9. Save model
joblib.dump(model, "models/model.pkl")

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel saved successfully!")


# 10. Get new input
quantity = float(input("\nEnter Quantity: "))
unit_price = float(input("Enter Unit Price: "))


# 11. Scale new input
new_data = scaler.transform([[quantity, unit_price]])


# 12. Predict revenue
prediction = model.predict(new_data)


print("Predicted Revenue:", round(prediction[0], 2))