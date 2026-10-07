import pandas as pd

# Load dataset
df = pd.read_csv("car data.csv")

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows where important values are missing
df = df.dropna(subset=["Selling_Price", "Present_Price", "Driven_kms"])

# Convert numerical columns to numeric
df["Selling_Price"] = pd.to_numeric(df["Selling_Price"], errors="coerce")
df["Present_Price"] = pd.to_numeric(df["Present_Price"], errors="coerce")
df["Driven_kms"] = pd.to_numeric(df["Driven_kms"], errors="coerce")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Remove rows that became invalid
df = df.dropna(subset=["Selling_Price", "Present_Price", "Driven_kms", "Year"])

# Display cleaned dataset information
print("\nCleaned Dataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nFirst 5 Rows:")
print(df.head())
# Create car age feature
current_year = 2026

df["Car_Age"] = current_year - df["Year"]

print("\nCar Age Feature:")
print(df[["Year", "Car_Age"]].head())
# Select features for machine learning
features = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Fuel_Type",
    "Selling_type",
    "Transmission",
    "Owner",
    "Car_Age"
]

target = "Selling_Price"

X = df[features]
y = df[target]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

print("\nFeature Data Types:")
print(X.dtypes)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Categorical columns
categorical_features = [
    "Fuel_Type",
    "Selling_type",
    "Transmission",
    "Owner"
]

# Numerical columns
numerical_features = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Car_Age"
]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Create Linear Regression model
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)

# Train the model
linear_model.fit(X_train, y_train)

# Make predictions
y_pred_linear = linear_model.predict(X_test)

print("\nLinear Regression model trained successfully!")
print("First 5 predictions:")
print(y_pred_linear[:5])
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred_linear)
mse = mean_squared_error(y_test, y_pred_linear)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_linear)

print("\n--- Linear Regression Evaluation ---")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
from sklearn.ensemble import RandomForestRegressor

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ]
)

random_forest_model.fit(X_train, y_train)

y_pred_rf = random_forest_model.predict(X_test)

print("\nRandom Forest model trained successfully!")
print("First 5 predictions:")
print(y_pred_rf[:5])
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n--- Random Forest Evaluation ---")
print("MAE :", mae_rf)
print("MSE :", mse_rf)
print("RMSE:", rmse_rf)
print("R2 Score:", r2_rf)
import matplotlib.pyplot as plt

models = ["Linear Regression", "Random Forest"]
r2_scores = [r2, r2_rf]

plt.figure(figsize=(8, 5))
plt.bar(models, r2_scores)

plt.title("Model Comparison - R2 Score")
plt.xlabel("Models")
plt.ylabel("R2 Score")

plt.ylim(0, 1)
plt.tight_layout()

plt.savefig("plots/model_comparison.png")
plt.show()

print("\nModel comparison graph saved successfully!")
import joblib

joblib.dump(linear_model, "car_price_model.pkl")

print("\nBest model saved successfully!")
# Example car price prediction

sample_car = pd.DataFrame({
    "Year": [2018],
    "Present_Price": [8.5],
    "Driven_kms": [30000],
    "Fuel_Type": ["Petrol"],
    "Selling_type": ["Dealer"],
    "Transmission": ["Manual"],
    "Owner": [0],
    "Car_Age": [8]
})

predicted_price = linear_model.predict(sample_car)

print("\n--- Car Price Prediction ---")
print("Predicted Selling Price:", round(predicted_price[0], 2), "lakhs")
# Example car price prediction

sample_car = pd.DataFrame({
    "Year": [2018],
    "Present_Price": [8.5],
    "Driven_kms": [30000],
    "Fuel_Type": ["Petrol"],
    "Selling_type": ["Dealer"],
    "Transmission": ["Manual"],
    "Owner": [0],
    "Car_Age": [8]
})

predicted_price = linear_model.predict(sample_car)

print("\n--- Car Price Prediction ---")
print("Predicted Selling Price:", round(predicted_price[0], 2), "lakhs")