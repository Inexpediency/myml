# # #
# At the end of testing, we can conclude that it is advisable to use RandomForestRegressor model.
# # #


import random
from model_comparison import take_all_graphs


def test_models(X, y):
    # # Model Training
    # Training sample
    training_count = random.randint(1, 600)
    X = X[:-training_count]
    y = y[:-training_count]
    # Testing sample
    X_test = X[-training_count:]
    y_test = y[-training_count:]
    # Testing
    take_all_graphs(X, y, X_test, y_test)  # All model on 1 graph
