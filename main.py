import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# Load data
data = pd.read_csv("C:\Users\Hemambika\OneDrive\Desktop\projects\temperature predictor\wells_for_prediction.csv")

# Specify columns
initial_well_logs_columns = ['NPHI', 'RHOB', 'GR', 'RT', 'PEF', 'CALI']
target_column = 'DT'

# Normalize well logs
scaler = MinMaxScaler(feature_range=(-1, 1))
data[initial_well_logs_columns] = scaler.fit_transform(data[initial_well_logs_columns])

# Set up features and target
X = data[initial_well_logs_columns]
y = data[target_column]

# Plot distributions of features
plt.figure(figsize=(15, 10))
for i, col in enumerate(initial_well_logs_columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data[initial_well_logs_columns + [target_column]].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (Linear Regression): {mse}')
print(f'R^2 Score (Linear Regression): {r2}')

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual DT')
plt.ylabel('Predicted DT')
plt.title('Actual vs Predicted DT')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r')
plt.show()

# Handle missing values
data = data.dropna()

# Create interaction terms
data['NPHI_GR'] = data['NPHI'] * data['GR']
data['RHOB_RT'] = data['RHOB'] * data['RT']

# Update features
well_logs_columns = ['NPHI', 'RHOB', 'GR', 'RT', 'PEF', 'CALI', 'NPHI_GR', 'RHOB_RT']
X = data[well_logs_columns]
y = data['DT']

# Split the data again with updated features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = RandomForestRegressor(random_state=42)

# Define hyperparameters for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (Random Forest): {mse}')
print(f'R^2 Score (Random Forest): {r2}')

# Cross-validate the best model
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='r2')
print(f'Cross-Validation R^2 Scores: {cv_scores}')
print(f'Mean Cross-Validation R^2 Score: {np.mean(cv_scores)}')

# Feature importances
importances = best_model.feature_importances_

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': well_logs_columns,
    'Importance': importances
})

# Sort and display feature importances
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importances:")
print(feature_importance_df)