import pandas
from helper_methods import *
import statsmodels.api as sm

data = pandas.read_csv('./project-data.csv')
development_data_raw = data[~data['SalePrice'].isnull()]
evaluation_data_raw = data[data['SalePrice'].isnull()]

development_data = development_data_raw.dropna()
evaluation_data = evaluation_data_raw.drop(columns=['SalePrice'])
evaluation_data.dropna()
categorical_feature_names = ['MSZoning', 'Street', 'Utilities', 'BldgType', 'BsmtQual', 'ExterQual', 'ExterCond', 'Heating', 'GarageCond']
development_data = label_categorical_features(development_data, categorical_feature_names)
evaluation_data = label_categorical_features(evaluation_data, categorical_feature_names)

# Part 2: Training a naive model
evaluation_data.drop(columns=['Id'], inplace=True)
development_data.drop(columns=['Id'], inplace=True)
test_errors = []
num_of_iterations = 500
for i in range(1, num_of_iterations):
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
    testing_predictions = naive_model.predict(testX)
    testing_error = mean_square_error(y_pred=testing_predictions, y_true=testY)

    test_errors.append(testing_error)

print('Avarage testing error of naive model = ', pandas.Series(test_errors).mean())