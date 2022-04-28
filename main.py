# Imports
import pandas as pd
from numpy import absolute
from numpy import mean
from numpy import std

# models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

# Constants
TEST_DATA_PERCENTAGE = 20  # Value in percentage %
USELESS_COLUMNS = ["id", "wind_speed", "atmo_opacity"]
ATTRIBUTE_COLUMNS = ["sol", "ls", "month", "terrestrial_month", "terrestrial_year", "terrestrial_quarter"]
GOAL_COLUMNS = ["min_temp", "max_temp", "pressure"]
MODELS_DICTIONARY = {
    "KNeighbors Regressor ": KNeighborsRegressor(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Neural Networks Regressor": MLPRegressor(random_state=1, max_iter=5000)
}

# Variables
dataFrame = pd.read_csv("mars-weather.csv")

# Clear the dataFrame
filteredData = dataFrame.drop(columns=USELESS_COLUMNS)
filteredData.dropna(inplace=True) # Remove incomplete data

# Convert the month column into an enum
filteredData["month"] = filteredData["month"].rank(method="dense", ascending=True).astype(int)

# Pre-process terrestrial date
filteredData["terrestrial_date"] = pd.to_datetime(filteredData["terrestrial_date"]) # Convert to dateTime64

date = filteredData["terrestrial_date"]
filteredData["terrestrial_month"] = date.dt.month
filteredData["terrestrial_year"] = date.dt.year
filteredData["terrestrial_quarter"] = date.dt.quarter

del filteredData["terrestrial_date"] # Drop the column as it won't be used any longer

X = filteredData.drop(columns=GOAL_COLUMNS)
Y = filteredData.drop(columns=ATTRIBUTE_COLUMNS)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_DATA_PERCENTAGE / 100)

# Evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)


def test_model(name, model_object):
    model_object.fit(X, Y)
    predictions = model_object.predict(X_test)
    n_scores = cross_val_score(model_object, X, Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    n_scores = absolute(n_scores)

    # summarize performance
    print("\nUsed algorithm: " + name)
    print("MAE: %.3f (%.3f)" % (mean(n_scores), std(n_scores)))


for model_name, model in MODELS_DICTIONARY.items():
    test_model(model_name, model)
