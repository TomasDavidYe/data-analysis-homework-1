from helper_methods import *
import statsmodels.api as sm


[development_data, evaluation_data] = get_data()
trueY = development_data['SalePrice']
naiveX = naive_transformation_of_features(development_data).drop(columns=['SalePrice'])
finalX = final_transformation_of_features(development_data).drop(columns=['SalePrice'])

naive_model = sm.OLS(exog=naiveX, endog=trueY).fit()
final_model = sm.OLS(exog=finalX, endog=trueY).fit()

naiveY = naive_model.predict(naiveX)
finalY = final_model.predict(finalX)

for i in range(0, 100):
    print('Index = ', i,  ' naive prediction = ', naiveY.values[i], ' true price = ', trueY.values[i],  ' final prediction = ', finalY.values[i])