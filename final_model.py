import pandas
from helper_methods import *
import statsmodels.api as sm

testing_errors = []
training_errors = []
num_of_iterations = 500


[development_data, evaluation_data] = getData()
development_data = final_transformation_of_features(development_data)

for i in range(0, num_of_iterations):
    print('Iteration ', i)

    # Splitting the development data randomly 80-20
    [trainX, trainY, testX, testY] = split_development_data(development_data)
    final_model = sm.OLS(exog=trainX, endog=trainY).fit()

    training_predictions = final_model.predict(trainX)
    training_error = root_mean_square_error(y_pred=training_predictions, y_true=trainY)
    training_errors.append(training_error)

    testing_predictions = final_model.predict(testX)
    testing_error = root_mean_square_error(y_pred=testing_predictions, y_true=testY)
    testing_errors.append(testing_error)

    if i == num_of_iterations - 1:
        print(final_model.summary())


print('Avarage training error of final model = ', pandas.Series(training_errors).mean())
print('Avarage testing error of final model  = ', pandas.Series(testing_errors).mean())
