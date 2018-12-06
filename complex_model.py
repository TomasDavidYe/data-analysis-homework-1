import pandas
from helper_methods import *
import statsmodels.api as sm
import math

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

[development_data, evaluation_data] = getData()

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

testing_errors = []


def get_column_names_from_indices(indices):
    result = [];
    for index in indices:
        result.append(all_column_names[index])
    return result


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

    all_column_names = list(trainX.columns.values)
    chosen_column_names = get_column_names_from_indices([0, 1, 2, 3, 6, 7, 8, 10])
    chosen_column_names
    column_names_for_log_transformation = get_column_names_from_indices([1, 6, 7, 8, 10])
    for column_name in column_names_for_log_transformation:
        trainX[column_name] = trainX[column_name].apply(lambda x: math.log2(x + 1))
        testX[column_name] = testX[column_name].apply(lambda x: math.log2(x + 1))

    trainX = trainX[chosen_column_names]
    testX = testX[chosen_column_names]

    # 0 1 2 3 6 7 8 10

    name_sets_for_given_transformation = {
        'Exp': get_column_names_from_indices([]),
        'Log': get_column_names_from_indices([2]),
        'Cube': get_column_names_from_indices([]),
        'Square': get_column_names_from_indices([]),
        'Root': get_column_names_from_indices([7, 8])
    }

    mixed_column_number_pairs = [[3,2], [1,3]]

    for number_pair in mixed_column_number_pairs:
        pair = get_column_names_from_indices(number_pair)
        column_name0 = pair[0]
        column_name1 = pair[1]
        trainX[column_name0 + 'Times' + column_name1] = trainX[column_name0]*trainX[column_name1]
        testX[column_name0 + 'Times' + column_name1] = testX[column_name0]*testX[column_name1]


    transformation_names = ['Exp', 'Log', 'Cube', 'Square', 'Root']
    exp = lambda x: math.exp(x)
    log = lambda x: math.log2(x + 1)
    cube = lambda x: x * x * x
    square = lambda x: x * x
    root = lambda x: math.sqrt(x)

    transformations = {
        'Exp': exp,
        'Log': log,
        'Cube': cube,
        'Square': square,
        'Root': root
    }

    for transformation_name in transformation_names:
        for column_name in name_sets_for_given_transformation[transformation_name]:
            transformation = transformations[transformation_name]
            trainX[column_name + transformation_name] = trainX[column_name].apply(transformation)
            testX[column_name + transformation_name] = testX[column_name].apply(transformation)

    new_model = sm.OLS(exog=trainX, endog=trainY).fit()
    if i == num_of_iterations - 1:
        print(new_model.summary())

    training_predictions = new_model.predict(trainX)
    training_error = root_mean_square_error(training_predictions, trainY)
    testing_predictions = new_model.predict(testX)
    testing_error = root_mean_square_error(testing_predictions, testY)
    testing_errors.append(testing_error)


print()
print()
print("Average error of new model = ", pandas.Series(testing_errors).mean())
