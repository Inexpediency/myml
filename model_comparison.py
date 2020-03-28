import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


def take_all_graphs(X, y, X_test, y_test):
    LinReg = LinearRegression()
    LinReg.fit(X, y)

    # n is 7 to have overview on whole week
    KNN = KNeighborsRegressor(n_neighbors=7)
    KNN.fit(X, y)

    Rid = Ridge()
    Rid.fit(X, y)

    DR = DummyRegressor(strategy='quantile', quantile=0.1)
    DR.fit(X, y)

    RFD = RandomForestRegressor(n_estimators=7)
    RFD.fit(X, y)

    # Neural Network
    # Create a MLPRegression model with effective parameters
    MLP = MLPRegressor(random_state=42, max_iter=1000, hidden_layer_sizes=(15, 35, 15), solver='lbfgs',
                       warm_start='True')  # max_fun didnt work
    MLP.fit(X, y)

    # # # Plot all graphs on 1 plot
    # # Models Predictions
    prediction1 = MLP.predict(X_test)
    prediction2 = RFD.predict(X_test)
    # prediction3 = DR.predict(X_test)  # DR is very bad
    prediction4 = Rid.predict(X_test)
    prediction5 = KNN.predict(X_test)
    # # Difference between the readings of models from real
    print("MLP", mean_absolute_error(prediction1[0], y_test.iloc[0]))  # Show MAE for 5 regression models
    print("RFR", mean_absolute_error(prediction2[0], y_test.iloc[0]))
    # print("DR", mean_absolute_error(prediction3[0], y_test.iloc[0]))
    print("Ridge", mean_absolute_error(prediction4[0], y_test.iloc[0]))
    print("KNN", mean_absolute_error(prediction5[0], y_test.iloc[0]))
    plt.plot(prediction1[0], label='Prediction MLP')  # Plot 5 regression models on the same plot
    plt.plot(prediction2[0], label='Prediction RFR')
    # plt.plot(prediction3[0], label='Prediction DR')
    plt.plot(prediction4[0], label='Prediction Ridge')
    plt.plot(prediction5[0], label='Prediction KNN')
    plt.plot(y_test.iloc[0], label='Real')
    plt.ylabel('Cost')
    plt.xlabel('Days')
    plt.legend()
    plt.show()
