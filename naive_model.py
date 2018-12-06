import pandas
from helper_methods import *
import statsmodels.api as sm

[development_data, evaluation_data] = getData()
testing_errors = []
training_errors = []

num_of_iterations = 500
for i in range(0, num_of_iterations):
    print('Iteration ', i)
    training_set = development_data.sample(frac=0.8)
    testing_set = development_data.drop(index=training_set.index)

    trainX = training_set.drop(columns=['SalePrice'])
    trainX = sm.add_constant(data=trainX, has_constant='add')
    trainY = training_set['SalePrice']
    testX = testing_set.drop(columns=['SalePrice'])
    testX = sm.add_constant(data=testX, has_constant='add')
    testY = testing_set['SalePrice']

    naive_model = sm.OLS(exog=trainX, endog=trainY).fit()

    training_predictions = naive_model.predict(trainX)
    training_error = mean_square_error(y_pred=training_predictions, y_true=trainY)
    training_errors.append(training_error)

    testing_predictions = naive_model.predict(testX)
    testing_error = mean_square_error(y_pred=testing_predictions, y_true=testY)
    testing_errors.append(testing_error)

    if i == num_of_iterations - 1:
        print(naive_model.summary())


print('Avarage training error of naive model = ', pandas.Series(training_errors).mean())
print('Avarage testing error of naive model  = ', pandas.Series(testing_errors).mean())
