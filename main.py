import random

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

#   Reading data frame from excel table
data_frame = pd.read_excel("data/usd_changing.xlsx")

# List of changing cost of usd
curs = data_frame.curs

past = 4 * 7  # Data for 2 weeks in last
future = 7  # Try predict curs for week in future

start = past  # Start point
end = len(curs) - future  # End point

# # # Creating new data frame # # #
new_df = []
for i in range(start, end):
    cols = curs[(i-past):(i+future)]
    new_df.append(list(cols))
past_columns = [f"Past_{p}" for p in range(past)]
future_columns = [f"Future_{f}" for f in range(future)]
transformed_df = pd.DataFrame(new_df, columns=(past_columns+future_columns))

X = transformed_df[past_columns]  # The part on which we study
y = transformed_df[future_columns]  # The part that we expect at the exit

# # # Comparison models
# from testing_models import test_models
# test_models(X, y)
# # # ~ Results
# # MLP 0.514357981146229
# # RFR 0.10857142857142321
# # Ridge 0.5581110567074015
# # KNN 0.3150102040816388
# # # -> Should use RFR

# # --- Take model params
from sklearn.model_selection import GridSearchCV
RFR = RandomForestRegressor()
param_grid = {
    'n_estimators': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'max_depth': [None, 3, 4, 5, 6, 7, 9, 15, 19, 27, 39, 50],
    'random_state': [20, 40, 60, 80]
}
# --- Grid Search Cross Validation with MAE scoring
GS = GridSearchCV(RFR, param_grid, scoring='neg_mean_absolute_error', cv=5)
# --- Fit Grid Search
GS.fit(X, y)

# --- Take results testing
print(GS.best_params_)  # Best model params from param_grid  ->  {'max_depth': 5, 'n_estimators': 13}
print(GS.best_score_)  # Best score == MAE  ->  -0.8182807519517048
model = GS.best_estimator_  # Take best fitted model

RFR = RandomForestRegressor(max_depth=5, n_estimators=13)
# RFR = RandomForestRegressor(n_estimators=7)
# RFR.fit(X, y)

# # Model Training
# Training sample
training_count = random.randint(1, 300)
X = X[:-training_count]
y = y[:-training_count]
# Testing sample
X_test = X[-training_count:]
y_test = y[-training_count:]
# Take Graph
RFR.fit(X, y)
prediction = RFR.predict(X_test)
plt.plot(prediction[0], label='Prediction')
plt.plot(y_test.iloc[0], label='Real')
plt.ylabel('Cost')
plt.xlabel('Days')
plt.legend()
plt.show()
