# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset for car price prediction from CSV file
# Replace 'car_price_dataset.csv' with the actual path to your CSV file
car_price_df = pd.read_csv('car_data.csv')

# Drop columns not needed for car price prediction
car_price_df.drop(columns=['Car_Name', 'Selling_type', 'Owner'], inplace=True)

# Perform one-hot encoding for categorical variables
car_price_df = pd.get_dummies(car_price_df, drop_first=True)

# Split the dataset into features (X) and target variable (y) for car price prediction
X_car_price = car_price_df.drop(columns=['Selling_Price'])  # Features
y_car_price = car_price_df['Selling_Price']  # Target variable

# Split the dataset into training and testing sets for car price prediction
X_train_car_price, X_test_car_price, y_train_car_price, y_test_car_price = train_test_split(X_car_price, y_car_price, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model for car price prediction
car_price_model = RandomForestRegressor(random_state=42)
car_price_model.fit(X_train_car_price, y_train_car_price)

# Make predictions on the testing set for car price prediction
y_pred_car_price = car_price_model.predict(X_test_car_price)

# Evaluate the car price prediction model
mse_car_price = mean_squared_error(y_test_car_price, y_pred_car_price)
r2_car_price = r2_score(y_test_car_price, y_pred_car_price)

print("Car Price Prediction Model Evaluation:")
print("Mean Squared Error:", mse_car_price)
print("R^2 Score:", r2_car_price)

# Now, proceed with the training and evaluation of the provided dataset model
# Load the provided dataset from CSV file
# Replace 'provided_dataset.csv' with the actual path to your CSV file
provided_df = pd.read_csv('car_data.csv')

# Perform one-hot encoding for categorical variables
provided_df = pd.get_dummies(provided_df, drop_first=True)

# Check the column names after one-hot encoding
print(provided_df.columns)

# Remove 'Car_Name' column if present in the dataset
if 'Car_Name' in provided_df.columns:
    provided_df.drop(columns=['Car_Name'], inplace=True)

# Split the dataset into features (X) and target variable (y) for the provided dataset
X_provided = provided_df.drop(columns=['Selling_Price'])  # Features
y_provided = provided_df['Selling_Price']  # Target variable

# Split the dataset into training and testing sets for the provided dataset
X_train_provided, X_test_provided, y_train_provided, y_test_provided = train_test_split(X_provided, y_provided, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model for the provided dataset
model = RandomForestRegressor(random_state=42)
model.fit(X_train_provided, y_train_provided)

# Make predictions on the testing set for the provided dataset
y_pred_provided = model.predict(X_test_provided)

# Evaluate the provided dataset model
mse_provided = mean_squared_error(y_test_provided, y_pred_provided)
r2_provided = r2_score(y_test_provided, y_pred_provided)

print("\nProvided Dataset Model Evaluation:")
print("Mean Squared Error:", mse_provided)
print("R^2 Score:", r2_provided)
