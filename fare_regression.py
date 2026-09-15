import math
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------------------------------------
# STEP 1: Load the data
# ---------------------------------------------------------
# Change this path if your train.csv is somewhere else in your notebook
data = pd.read_csv("titanic/train.csv")

# ---------------------------------------------------------
# STEP 2: Pick our variables
# ---------------------------------------------------------
# Dependent variable (what we want to predict): Fare
# Independent variables (what we use to predict it):
#   - Pclass: ticket class (1st, 2nd, 3rd)
#   - Age: passenger age
#   - SibSp: number of siblings/spouses aboard
data = data[["Fare", "Pclass", "Age", "SibSp"]]

# Some passengers are missing an Age value, so drop those rows
data = data.dropna()

# X = independent variables, y = dependent variable
X = data[["Pclass", "Age", "SibSp"]]
y = data["Fare"]

# ---------------------------------------------------------
# STEP 3: Split into training data and testing data
# ---------------------------------------------------------
# We train the model on 80% of the data and test it on the other 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# STEP 4: Create and train the linear regression model
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------------
# STEP 5: Use the model to make predictions on the test data
# ---------------------------------------------------------
y_pred = model.predict(X_test)

# ---------------------------------------------------------
# STEP 6: Look at the results
# ---------------------------------------------------------
print("Intercept:", model.intercept_)
print("Coefficient for Pclass:", model.coef_[0])
print("Coefficient for Age:", model.coef_[1])
print("Coefficient for SibSp:", model.coef_[2])

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

print("R-squared:", r2)
print("RMSE:", rmse)
