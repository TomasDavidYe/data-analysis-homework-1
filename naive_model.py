import pandas
from helper_methods import *
import statsmodels.api as sm

[development_data, evaluation_data] = getData()
testing_errors = []
training_errors = []

num_of_iterations = 500
for i in range(0, num_of_iterations):
    print('Iteration ', i)

    # Splitting the development data randomly 80-20
    [trainX, trainY, testX, testY] = split_development_data(development_data)
    trainX = naive_transformation(trainX)
    testX = naive_transformation(testX)
    naive_model = sm.OLS(exog=trainX, endog=trainY).fit()

    training_predictions = naive_model.predict(trainX)
    training_error = root_mean_square_error(y_pred=training_predictions, y_true=trainY)
    training_errors.append(training_error)

    testing_predictions = naive_model.predict(testX)
    testing_error = root_mean_square_error(y_pred=testing_predictions, y_true=testY)
    testing_errors.append(testing_error)

    if i == num_of_iterations - 1:
        print(naive_model.summary())


print('Avarage training error of naive model = ', pandas.Series(training_errors).mean())
print('Avarage testing error of naive model  = ', pandas.Series(testing_errors).mean())
