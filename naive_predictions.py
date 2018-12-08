from helper_methods import *
import statsmodels.api as sm


[development_data, evaluation_data] = get_data()
Y = development_data['SalePrice']
naiveX = naive_transformation_of_features(development_data).drop(columns=['SalePrice'])
evalX = naive_transformation_of_features(evaluation_data)


naive_model = sm.OLS(exog=naiveX, endog=Y).fit()
naive_predictions = naive_model.predict(evalX)
naive_predictions.to_csv(path='./output/naive_predictions.csv')
indices = Y.index
