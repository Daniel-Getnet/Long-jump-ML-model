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
gb = GradientBoostingRegressor(
        n_estimators= 100, # [100, 500] step of 1
        learning_rate= 0.08,  # [0.01, 0.3] step 0f 0.01
        min_samples_split= 10,  # [2, 10] step 0f 1
        subsample= 0.6,   # (0.1, 1.0] step of 0.1
        random_state=42  # Seed for the random number generator
    )

# Fit to training set
gb.fit(train_X, train_y),

# Predict on test set
pred_y = gb.predict(test_X)

# Compute feature importance
feature_importance = gb.feature_importances_

# Get feature names
feature_names = dani_file.columns[:-1]

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)


# Scatter plot of test_y against test predictions
plt.scatter(test_y, pred_y)
m, b = np.polyfit(test_y, pred_y, 1)
plt.plot(test_y, m*test_y + b, label='fit')
r2_test = r2_score(test_y, pred_y)
plt.title(f'R\u00b2 = {r2_test:.4f}')
plt.xlabel('Actual (m)')
plt.ylabel('Predicted (m)')
plt.legend()
plt.show()


# Plot feature importance
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.xticks(fontfamily='Calibri')
plt.yticks(fontfamily='Calibri')
plt.gca().invert_yaxis()
plt.show()

# Model evaluation 
print(f"r_square_for_the_test_dataset : {r2_score(test_y, pred_y):.4f}") 
print(f"mean_absolute_error : {mean_absolute_error(test_y, pred_y):.4f}")
print(f"root_mean_squared_error : {mean_squared_error(test_y, pred_y, squared=False):.4f}")