import pandas as pd
import numpy as np
import statsmodels.api as sm
import math


def getData():
    data = pd.read_csv('./project-data.csv')
    development_data_raw = data[~data['SalePrice'].isnull()]
    evaluation_data_raw = data[data['SalePrice'].isnull()]

    development_data = development_data_raw.dropna()
    evaluation_data = evaluation_data_raw.drop(columns=['SalePrice'])
    evaluation_data.dropna()
    categorical_feature_names = ['MSZoning', 'Street', 'Utilities', 'BldgType', 'BsmtQual', 'ExterQual', 'ExterCond',
                                 'Heating', 'GarageCond']
    development_data = label_categorical_features(development_data, categorical_feature_names)
    evaluation_data = label_categorical_features(evaluation_data, categorical_feature_names)

    evaluation_data.drop(columns=['Id'], inplace=True)
    development_data.drop(columns=['Id'], inplace=True)
    return [development_data, evaluation_data]


def split_development_data(development_data):
    training_set = development_data.sample(frac=0.8)
    testing_set = development_data.drop(index=training_set.index)
    trainX = training_set.drop(columns=['SalePrice'])
    trainY = training_set['SalePrice']
    testX = testing_set.drop(columns=['SalePrice'])
    testY = testing_set['SalePrice']
    return [trainX, trainY, testX, testY]


def label_categorical_features(data_set, column_names):
    return pd.get_dummies(data=data_set, prefix=column_names)


def root_mean_square_error(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise Exception('Vectors do not have the same type')
    diff = y_pred - y_true
    return math.sqrt(diff.T.dot(diff)/len(diff))


def naive_transformation_of_features(data):
    return sm.add_constant(data, has_constant='add')


def final_transformation_of_features(data: pd.DataFrame) -> pd.DataFrame:
    final_selected_column_names = ['LotArea', 'OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'GarageCars', 'ExterQual_TA']
    all_column_names = list(data.columns.values)
    dropable_column_names = difference_between_lists(original=all_column_names, removed=final_selected_column_names)

    if 'SalePrice' in dropable_column_names:
        dropable_column_names.remove('SalePrice')
    data.drop(columns=dropable_column_names, inplace=True)
    data = transform_data(data)
    return sm.add_constant(data)


def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    result: pd.DataFrame = data.copy()

    column_names_for_log_transformation = ['LotArea', '1stFlrSF', '2ndFlrSF', 'GarageCars']
    column_names_for_adding_log_transformation = ['OverallQual']
    log = lambda x: math.log2(x + 1)
    for column_name in column_names_for_log_transformation:
        result[column_name] = result[column_name].apply(log)

    for column_name in column_names_for_adding_log_transformation:
        result[column_name + 'Log'] = result[column_name].apply(log)

    column_names_for_adding_quadratic_transformation = ['LotArea', 'OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'GarageCars']
    square = lambda x: x*x
    for column_name in column_names_for_adding_quadratic_transformation:
        result[column_name + 'Quad'] = result[column_name].apply(square)

    column_names_for_adding_root_transformation =['2ndFlrSF', 'GarageCars']
    root = lambda x: math.sqrt(x)
    for column_name in column_names_for_adding_root_transformation:
        result[column_name + 'Root'] = result[column_name].apply(root)

    pairs_of_column_names_for_adding_product_features = [['OverallQual', 'OverallCond'], ['LotArea', 'OverallCond']]
    for name_pair in pairs_of_column_names_for_adding_product_features:
        column_name_0 = name_pair[0]
        column_name_1 = name_pair[1]
        result[column_name_0 + 'Times' + column_name_1] = result[column_name_0] * result[column_name_1]

    return result

def difference_between_lists(original, removed):
    return list(set(original) - set(removed))
