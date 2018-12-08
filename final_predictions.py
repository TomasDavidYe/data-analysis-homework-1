from helper_methods import *
import statsmodels.api as sm


[development_data, evaluation_data] = get_data()
Y = development_data['SalePrice']
naiveX = final_transformation_of_features(development_data).drop(columns=['SalePrice'])
evalX = final_transformation_of_features(evaluation_data)


final_model = sm.OLS(exog=naiveX, endog=Y).fit()
final_predictions = final_model.predict(evalX)
final_predictions.to_csv(path='./output/final_predictions.csv')
