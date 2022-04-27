# Imports
import math
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# Constants
TEST_DATA_PERCENTAGE = 20  # Value in percentage %
USELESS_COLUMNS = ["id", "wind_speed", "atmo_opacity"]
ATTRIBUTE_COLUMNS = ["terrestrial_date", "sol", "ls", "month"]
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

X = filteredData.drop(columns=GOAL_COLUMNS)
Y = filteredData.drop(columns=ATTRIBUTE_COLUMNS)


model = KNeighborsRegressor()

#model.fit(X, Y)
