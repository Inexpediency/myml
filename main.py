import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# from testing_models import test_models

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
# test_models(X, y)
# # # ~ Results
# # MLP 0.514357981146229
# # RFR 0.10857142857142321
# # Ridge 0.5581110567074015
# # KNN 0.3150102040816388
# # # -> Should use RFR

clf = RandomForestRegressor(n_estimators=7)
clf.fit(X, y)
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=7, n_jobs=1,
                      oob_score=False, random_state=None, verbose=0, warm_start=False)


