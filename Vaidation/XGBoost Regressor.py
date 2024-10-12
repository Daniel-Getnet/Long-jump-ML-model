import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Importing dataset
dani_file = pd.read_csv('MLLJ.csv')

# Create a separate LabelEncoder object for each categorical column
dani_gender = LabelEncoder()

# Fit the label encoder and transform each categorical column individually
dani_file["Gender"] = dani_gender.fit_transform(dani_file["Gender"])

# Handling missing values by replacing them with the mean of each feature
X = dani_file.iloc[:, :-1]
X.fillna(X.mean(), inplace=True) 

y = dani_file.iloc[:, -1]
y.fillna(y.mean(), inplace=True)  

# Splitting dataset
train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                    test_size = 0.25, 
                                                    random_state = 42)
# Instantiate Random Forest Regressor
xgb_regressor = XGBRegressor(
       n_estimators=375,  # [100, 500] step of 1
        learning_rate=0.28,  # [0.01, 0.3] step of 0.01
        max_depth=2,  # [1, 10] step of 1
        subsample=0.89,   # [0.5, 1] step of 0.1
        random_state=42  # Seed for the random number generator

)

# Fit to training set
xgb_regressor.fit(train_X, train_y)

# Predict on test set
pred_y = xgb_regressor.predict(test_X)

# Model evaluation 
print(f"r_square_for_the_test_dataset : {r2_score(test_y, pred_y):.4f}") 
print(f"mean_absolute_error : {mean_absolute_error(test_y, pred_y):.4f}")
print(f"root_mean_squared_error : {mean_squared_error(test_y, pred_y, squared=False):.4f}")

print('**************************************')

# Initialize LabelEncoders for categorical variables
le_dani_gender = LabelEncoder()

# Fit the LabelEncoders with possible categories
possible_dani_gender = ['F', 'M']

# Fit the LabelEncoders with possible categories
le_dani_gender.fit(possible_dani_gender)

# Function to prompt user for input with handling missing values
def prompt_user_input(prompt):
    while True:
        user_input = input(prompt)
        if user_input.strip():  # Check if input is not empty after stripping whitespace
            return user_input
        else:
            print("Missing value detected. Filling with mean value.")
            return np.nan  # Return NaN for missing values

# Prompt the user for input for each feature
gender = prompt_user_input ("Enter the gender: ")
take_off_loos = float(prompt_user_input("Enter take off loss: "))
third_last_step_length = float(prompt_user_input("Enter third last step length: "))
third_last_step_hor_veocity_bto = float(prompt_user_input("Enter 3rd last step horizontal Velocity BTO: "))
second_last_step_hor_veocity_bto = float(prompt_user_input("Enter 2rd last step horizontal Velocity BTO: "))
last_step_hor_veocity_bto = float(prompt_user_input("Enter last step horizontal Velocity BTO: "))
hr_velocity_take_off = float(prompt_user_input("Enter horizontal velocity at take off : "))
vr_velocity_take_off = float(prompt_user_input("Enter vertical velocity at take off: "))
third_last_step_contact_time = float(prompt_user_input("Enter 3rd last step Contact time: "))
third_last_step_flight_time = float(prompt_user_input("Enter 3rd last step flight time: "))
second_last_step_contact_time  = float(prompt_user_input("Enter 2nd last step Contact time: "))
trunk_angle_to = float(prompt_user_input("Enter trunk angle at to: "))
body_inclination_angle_at_td = float(prompt_user_input("Enter Body inclination angle at TD: "))
mean_knee_angular_velocity_td = float(prompt_user_input("Enter mean knee angular velocity TD: "))
thigh_angular_velocity_of_swing_leg_at_TO = float(input("Enter thigh angular velocity of swing leg at TO: "))

# Create a dictionary with the provided values
data = {
    'Gender': [gender],
    'Take off loss': [take_off_loos],
    '3rd last Step Length ': [third_last_step_length],
    '3rd Last step horizontal Velocity BTO (m/s)': [third_last_step_hor_veocity_bto],
    '2nd Last step horizontal velocity BTO (m/s)': [second_last_step_hor_veocity_bto],
    'Last step horizontal Velocity BTO (m/s)': [last_step_hor_veocity_bto],
    'Horizontal velocity at take off (m/s)': [hr_velocity_take_off],
    'Vertical velocity at take off (m/s)': [vr_velocity_take_off],
    '3rd last step Contact time (s)': [third_last_step_contact_time],
    '3rd last step flight time (s)': [third_last_step_flight_time],
    '2nd Last step contact time (s)': [second_last_step_contact_time],
    'Trunk angle at T0': [trunk_angle_to],
    'Body inclination angle at TD': [body_inclination_angle_at_td],
    'mean knee angular velocity TD': [mean_knee_angular_velocity_td],
    'Thigh angular velocity of swing leg at TO': [thigh_angular_velocity_of_swing_leg_at_TO]  # Correct variable name
}

# Create the DataFrame
new_data = pd.DataFrame(data)

# Handle missing values by replacing them with the mean of each feature
new_data.fillna(X.mean(), inplace=True)

# If there are categorical variables, encode them
new_data["Gender"] = le_dani_gender.transform(new_data["Gender"])

# Predict on new data
new_predictions = xgb_regressor.predict(new_data)

print('**************************************')

print("The estimated official distace is:", new_predictions)
