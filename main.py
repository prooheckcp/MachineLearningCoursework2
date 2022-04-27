# Imports
import math
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score

# Constants
TEST_DATA_PERCENTAGE = 20  # Value in percentage %
USELESS_COLUMNS = ["id", "wind_speed", "atmo_opacity"]
ATTRIBUTE_COLUMNS = ["terrestrial_date", "sol", "ls", "month"]
GOAL_COLUMNS = ["min_temp", "max_temp", "pressure"]
MODELS_DICTIONARY = {
    "Logistic regression model": LogisticRegression(max_iter=math.inf)
}

# Variables
dataFrame = pd.read_csv("mars-weather.csv")

# Clear the dataFrame
filteredData = dataFrame.drop(columns=USELESS_COLUMNS)
filteredData.dropna(inplace=True) # Remove incomplete data

# Convert the month column into an enum
months_mapping = {k: v for k, v in enumerate(filteredData.month.unique())}
filteredData["month"] = filteredData.month.map(months_mapping)

X = filteredData.drop(columns=GOAL_COLUMNS)
Y = filteredData.drop(columns=ATTRIBUTE_COLUMNS)

print(X.shape, Y.shape)

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_DATA_PERCENTAGE / 100)


def testing_model(model):
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    return accuracy_score(Y_test, predictions)


#for key, model_instance in MODELS_DICTIONARY.items():
    #testing_model(model_instance)

