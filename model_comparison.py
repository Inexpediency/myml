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
    prediction = LinReg.predict(X_test)

    # n is 7 to have overview on whole week
    KNN = KNeighborsRegressor(n_neighbors=7)
    KNN.fit(X, y)
    KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                        metric_params=None, n_jobs=1, n_neighbors=7, p=2,
                        weights='uniform')

    Rid = Ridge()
    Rid.fit(X, y)
    Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
          normalize=False, random_state=None, solver='auto', tol=0.001)

    DR = DummyRegressor(strategy='quantile', quantile=0.1)
    DR.fit(X, y)
    DummyRegressor(constant=None, quantile=0.1, strategy='quantile')

    RFD = RandomForestRegressor(n_estimators=7)
    RFD.fit(X, y)
    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                          max_features='auto', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=7, n_jobs=1,
                          oob_score=False, random_state=None, verbose=0, warm_start=False)

    # Neural Network
    # Create a MLPRegression model with effective parameters
    MLP = MLPRegressor(random_state=42, max_iter=1000, hidden_layer_sizes=(15, 35, 15), solver='lbfgs',
                       warm_start='True')  # max_fun didnt work
    MLP.fit(X, y)
    MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
                 beta_2=0.999, early_stopping=False, epsilon=1e-08,
                 hidden_layer_sizes=(15, 35, 15), learning_rate='constant',
                 learning_rate_init=0.001, max_iter=1000, momentum=0.9,
                 nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,
                 solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                 warm_start='True')

    # Plot all graphs on 1 plot
    prediction1 = MLP.predict(X_test)
    prediction2 = RFD.predict(X_test)
    # prediction3 = DR.predict(X_test)  # DR is very bad
    prediction4 = Rid.predict(X_test)
    prediction5 = KNN.predict(X_test)
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
