# Author: Kasper Seglem

import pandas as pd

# Load the data
data = pd.read_csv("dropout.csv", delimiter=";")

# Check for missing values
missing_values = data.isnull().sum()

missing_values

# Separate target from predictors
X = data.drop("Target", axis=1)
y = data["Target"]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical variables
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

from sklearn.linear_model import LogisticRegression
import numpy as np

# Train a logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Get feature importances
# For logistic regression, the magnitude of the coefficients can be interpreted as feature importance
feature_importances = np.abs(model.coef_).flatten()

# Pair the feature names with their importance values
feature_importance_dict = dict(zip(X.columns, feature_importances))

# Sort the features based on importance
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# Printing the results in a neat format
print("\nVariables ranked by their importance in predicting student dropout:\n")
for rank, (feature, importance) in enumerate(sorted_features, 1):
    print(f"{rank}. {feature}: {importance:.4f}")
print("\n")

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the figure size and grid layout
plt.figure(figsize=(20, 15))
sns.set_style("whitegrid")

# Plotting histograms for the numerical columns
for index, column in enumerate(numerical_cols, 1):
    plt.subplot(6, 6, index)
    sns.histplot(data[column], kde=True)
    plt.title(f"Distribution of {column}")
    plt.tight_layout()

plt.show()


# Selected influential variables for visualization
selected_vars = [
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 1st sem (approved)',
    'Tuition fees up to date',
    'Curricular units 2nd sem (grade)'
]

# Plotting pairwise scatter plots
sns.pairplot(data[selected_vars])
plt.suptitle("Pairwise scatter plots of influential variables", y=1.02)
plt.tight_layout()
plt.show()

# Generating a correlation matrix for the selected variables
correlation_matrix = data[selected_vars].corr()

# Plotting the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix of Influential Variables")
plt.show()
