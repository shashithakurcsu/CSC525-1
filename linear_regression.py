# Linear Regression assignment using California Housing dataset
# CSU Global Campus
# Principles of Machine Learning - CSC525-1 Module 3

# Import all libiries needed for this assignment
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import pandas.plotting as pd_plotting
from pandas.plotting import scatter_matrix
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns


# Step 1: Load California housing data.
# Note - Although as per instructions, were suppposed to Boston housing data,
# The Boston housing prices dataset has an ethical problem ans it has been deprecated 
# and removed from sklearn datasets.
# Therefore, for this assignment, I am using California housing data.

housing = fetch_california_housing(as_frame=True)
# Step 2: Lets take a look at the data.
housing = housing.frame
# Step 3: Display the first 5 rows of the data to see what it looks like
print("California Housing Dataset:")
print(housing.head()) 

# Step 4: Before linear regression, lets visualize the data by creating histograms
housing.hist(bins=50, figsize=(20,15))
plt.show()

# Step 5: compute the pair wise correlation for all columns
correlation_matrix = housing.corr()
print("Correlation Matrix:")
print(correlation_matrix)
# Create a heatmap to visualize the correlation matrix

sns.heatmap(data=correlation_matrix, annot=True)
plt.title("Correlation Matrix Heatmap")
plt.show()

# The best correlation is between 'MedInc' and 'MedHouseVal'
# Step 6: Create a scatter plot to visualize the relationship between 'MedInc' and 'MedHouseVal'
plt.figure(figsize=(10, 6))
plt.scatter(housing['MedInc'], housing['MedHouseVal'], alpha=0.5)
plt.title('Scatter Plot of MedInc vs MedHouseVal')
plt.xlabel('Median Income (MedInc)')
plt.ylabel('Median House Value (MedHouseVal)')
plt.grid(True)
plt.show()

# Not creating a scatter matrix as it is not very useful based in the correlation matrix
# Step 7: Let us split the data into training and testing sets. Randonly take 20% of the data for testing.
#X = housing[['MedInc']]
#X = housing.drop('MedHouseVal', axis=1)
X = housing[['MedInc', 'AveRooms', 'Latitude', 'Longitude']]
y = housing['MedHouseVal']
X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.2, random_state=42)   

# Step 8: Create a linear regression model using sklearn
mymodel = Pipeline([
    ('scaler', StandardScaler()),  # Standardize the features
    ('linear_regression', LinearRegression())  # Linear Regression model
])

# Step 9: Next step is to fit the model to the training data
mymodel.fit(X_training, y_training)

# Step 10: Now use the tained model to make predictions on the testing data
y_predicted = mymodel.predict(X_testing)
# Step 11: Calculate the R-squared value to evaluate the model's performance
r_squared = r2_score(y_testing, y_predicted)
print(f"Model Performance: R-squared value: {r_squared:.4f}")

# Step 12: Visualize the predictions against the actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_testing, y_predicted, alpha=0.5)
plt.plot([y_testing.min(), y_testing.max()], [y_testing.min(), y_testing.max()], color='red', linestyle='--')
plt.title('Predicted vs Actual Values')
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.grid(True)
plt.show()

# Step 13: Display the coefficients of the linear regression model
coefficients = mymodel.named_steps['linear_regression'].coef_
intercept = mymodel.named_steps['linear_regression'].intercept_
print("Coefficients of the Linear Regression Model:")
for feature, coef in zip(X.columns, coefficients):
    print(f"{feature}: {coef:.4f}")
print(f"Intercept: {intercept:.4f}")


