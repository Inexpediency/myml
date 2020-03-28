import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random

from model_comparison import take_all_graphs

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

# # # Model Training
# # Training sample
# training_count = random.randint(1, 600)
# X = X[:-training_count]
# y = y[:-training_count]
# # Testing sample
# X_test = X[-training_count:]
# y_test = y[-training_count:]
# # Testing
# take_all_graphs(X, y, X_test, y_test)  # All model on 1 graph

# # #
# At the end of testing, we can conclude that it is advisable to use RandomForestRegressor model.
# # #


