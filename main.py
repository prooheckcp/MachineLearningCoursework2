# Imports
import math
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Constants
TEST_DATA_PERCENTAGE = 20  # Value in percentage %
USELESS_COLUMNS = ["id", "wind_speed", "atmo_opacity"]
ATTRIBUTE_COLUMNS = ["sol", "ls", "month", "terrestrial_month", "terrestrial_year", "terrestrial_quarter"]
GOAL_COLUMNS = ["min_temp", "max_temp", "pressure"]
MODELS_DICTIONARY = {
    "Logistic regression model": KNeighborsRegressor()
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

model = KNeighborsRegressor()

model.fit(X, Y)
