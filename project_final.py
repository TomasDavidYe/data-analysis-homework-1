import pandas
from helper_methods import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
from sklearn.metrics import mean_squared_error

# The script will be divided into four parts:
# 1) Loading and pre-processing data
# 2) Trying a naive model
# 3) Observing the data
# 4) Training a better model


# Part 1: Loading data
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

training_set = development_data.sample(frac=0.8)
testing_set = development_data.drop(index=training_set.index)


trainX = training_set.drop(columns=['SalePrice'])
trainX = sm.add_constant(data=trainX, has_constant='add')
trainY = training_set['SalePrice']
testX = testing_set.drop(columns=['SalePrice'])
testX = sm.add_constant(data=testX, has_constant='add')
testY = testing_set['SalePrice']



naive_model = sm.OLS(exog=trainX, endog=trainY).fit()
print("Naive model summary")
print(naive_model.summary())  # The model matrix is close to singular
print()
print()
print()


# Calculating training error
training_predictions = naive_model.predict(trainX)
training_error = mean_square_error(y_pred=training_predictions, y_true=trainY)
print("Training error of Naive model = ", training_error)  # The error is ~1000 with small variance


# training_residuals = training_predictions - trainY
# all_column_names = list(trainX.columns.values)
# all_column_names
# plt.close()
# column_name = all_column_names[3]
# plt.scatter(x=trainX[column_name], y=training_residuals)


# Calculating testing error
testing_predictions = naive_model.predict(testX)
testing_error = mean_square_error(y_pred=testing_predictions, y_true=testY)
print(" Testing error of Naive model = ", testing_error)  # The error is ~2000 with big variance (Depending on the splitting)

# Let us try to find a more consistent model


# Part 3: Observing the data
def print_category_status():
    for categorical_feature_name in categorical_feature_names:
        print(categorical_feature_name)
        print(development_data_raw[categorical_feature_name].value_counts())
        print()


# Drop categories Street, Utilities, Heating, Garage cond since they are too one-sided
# Merge smaller categories in the rest categorical columns to make the situation binary

dropped_column_names = ['Street', 'Utilities', 'GarageCond', 'Heating']
for name in dropped_column_names:
    categorical_feature_names.remove(name)
development_data_raw.drop(columns=dropped_column_names, inplace=True)
development_data_raw.replace({'BsmtQual': {'Gd': 'GdExFa','Ex': 'GdExFa', 'Fa': 'GdExFa'}}, inplace=True)
development_data_raw.replace({'ExterCond': {'Gd': 'GdFaExPo', 'Fa': 'GdFaExPo', 'Ex': 'GdFaExPo', 'Po': 'GdFaExPo'}}, inplace=True)
development_data_raw.replace({'ExterQual': {'Gd': 'GdExFa', 'Ex': 'GdExFa', 'Fa': 'GdExFa'}}, inplace=True)
development_data_raw.replace({'BldgType': {'TwnhsE': 'TDZ2', 'Duplex': 'TDZ2', 'Twnhs': 'TDZ2', '2fmCon': 'TDZ2'}}, inplace=True)
development_data_raw.replace({'MSZoning': {'RM': 'RFRC', 'FV': 'RFRC', 'RH': 'RFRC', 'C (all)': 'RFRC'}}, inplace=True)
development_data = label_categorical_features(development_data_raw, categorical_feature_names)
development_data.drop(inplace=True, columns=['MSZoning_RL', 'BldgType_1Fam', 'BsmtQual_GdExFa', 'ExterQual_GdExFa', 'ExterCond_GdFaExPo'])
development_data.drop(columns=['MSSubClass', 'BldgType_TDZ2'], inplace=True)
development_data.drop(columns=['ExterCond_TA'], inplace=True)
development_data.drop(columns=['BsmtQual_TA'], inplace=True)
development_data.drop(columns=['Id'], inplace=True)



# 4) Training a better model
training_set = development_data.sample(frac=0.8)
testing_set = development_data.drop(index=training_set.index)


trainX = training_set.drop(columns=['SalePrice'])
trainX = sm.add_constant(data=trainX, has_constant='add')
trainY = training_set['SalePrice']
testX = testing_set.drop(columns=['SalePrice'])
testX = sm.add_constant(data=testX, has_constant='add')
testY = testing_set['SalePrice']

def get_column_names_from_indices(indices):
    result = [];
    for index in indices:
        result.append(all_column_names[index])
    return result


all_column_names = list(trainX.columns.values)
chosen_column_names = get_column_names_from_indices([0, 1, 2, 3, 6, 7, 8, 10])
column_names_for_log_transformation = get_column_names_from_indices([2, 3])

trainX = trainX[chosen_column_names]
testX = testX[chosen_column_names]


name_sets_for_given_transformation = {
    'exp': get_column_names_from_indices([]),
    'log': get_column_names_from_indices([]),
    'cube': get_column_names_from_indices([]),
    'square': get_column_names_from_indices([]),
    'root': get_column_names_from_indices([])
}


transformation_names = ['exp', 'log', 'cube', 'square', 'root']
exp = lambda x: math.exp(x)
log = lambda x: math.log2(x + 1)
cube = lambda x: x*x*x
square = lambda x: x*x
root = lambda x: math.sqrt(x)

transformations = {
    'exp': exp,
    'log': log,
    'cube': cube,
    'square': square,
    'root': root
}


for transformation_name in transformation_names:
    for column_name in name_sets_for_given_transformation[transformation_name]:
        transformation = transformations[transformation_name]
        trainX[column_name + transformation_name] = trainX[column_name].apply(transformation)

# chosen_column_names_for_squaring = get_column_names_from_indices([])
# chosen_column_names_for_transformation = get_column_names_from_indices([1, 6, 7])
#



# for column_name in chosen_column_names_for_transformation:
#     trainX[column_name] = trainX[column_name].apply(transformation)
#     testX[column_name] = testX[column_name].apply(transformation)
#
# for column_name in chosen_column_names_for_squaring:
#     trainX[column_name + 'Squared'] = trainX[column_name]*trainX[column_name]*trainX[column_name]
#     testX[column_name + 'Squared'] = testX[column_name]*testX[column_name]*testX[column_name]

# trainX['OverallQual'] = trainX['OverallQual'].apply(lambda x: -x*x)
# testX['OverallQual'] = testX['OverallQual'].apply(lambda x: -x*x)


new_model = sm.OLS(exog=trainX, endog=trainY).fit()
print(new_model.summary())


# Calculating training error
training_predictions = new_model.predict(trainX)
training_error = mean_square_error(training_predictions, trainY)
print("Training error of New model = ", training_error)


def plot(i):
    plt.close()
    training_residuals = training_predictions - trainY
    column_name = chosen_column_names[i]
    print(column_name)
    plt.scatter(x=trainX[column_name], y=training_residuals)

# plot(2)

# for i in range (1, len(trainX.columns)):
#     plot(i)
#     print('showing plot no ', i)


#plot(2)
#from analysing plots we conclude that only feature ''



# Calculating testing error
testing_predictions = new_model.predict(testX)
testing_error = mean_square_error(testing_predictions, testY)
print(" Testing error of New model = ", testing_error)

#
# A = trainX['LotArea']
# B = A*A
# B.rename('LotAreaSquared')
# trainX['LotAreaSquared'] = B
#
#
