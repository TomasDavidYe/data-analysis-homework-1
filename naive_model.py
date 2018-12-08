import pandas
from helper_methods import *
import statsmodels.api as sm

testing_costs = []
testing_errors = []
training_costs = []
training_errors = []
num_of_iterations = 1000

[development_data, evaluation_data] = get_data()
development_data = naive_transformation_of_features(development_data)

for i in range(0, num_of_iterations):
    print('Iteration ', i)

    # Splitting the development data randomly 80-20
    [trainX, trainY, testX, testY] = split_development_data(development_data)
    naive_model = sm.OLS(exog=trainX, endog=trainY).fit()

    training_predictions = naive_model.predict(trainX)
    training_cost = root_mean_square_error(y_pred=training_predictions, y_true=trainY)
    training_costs.append(training_cost)
    training_error = error(y_pred=training_predictions, y_true=trainY)
    training_errors.append(training_error)

    testing_predictions = naive_model.predict(testX)
    testing_cost = root_mean_square_error(y_pred=testing_predictions, y_true=testY)
    testing_costs.append(testing_cost)
    testing_error = error(y_pred=testing_predictions, y_true=testY)
    testing_errors.append(testing_error)

    if i == num_of_iterations - 1:
        print(naive_model.summary())


print('Avarage training cost of naive model = ', pandas.Series(training_costs).mean())
print('Avarage testing cost of naive model  = ', pandas.Series(testing_costs).mean())
print('Avarage training error of naive model = ', pandas.Series(training_errors).mean())
print('Avarage testing error of naive model  = ', pandas.Series(testing_errors).mean())
