# Imports
import pandas as pd
from sklearn.model_selection import train_test_split

# Constants
TEST_DATA_PERCENTAGE = 20  # Value in percentage %
USELESS_COLUMNS = ["id", "wind_speed", "atmo_opacity"]
ATTRIBUTE_COLUMNS = ["terrestrial_date", "sol", "ls", "month"]
GOAL_COLUMNS = ["min_temp", "max_temp", "pressure"]

# Variables
dataFrame = pd.read_csv("mars-weather.csv")

# Clear the dataFrame
filteredData = dataFrame.drop(columns=USELESS_COLUMNS)
filteredData.dropna(inplace=True) # Remove incomplete data

X = filteredData.drop(columns=GOAL_COLUMNS)
Y = filteredData.drop(columns=ATTRIBUTE_COLUMNS)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_DATA_PERCENTAGE / 100)