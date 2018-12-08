import pandas as pd
import statsmodels.api as sm
import math


def get_data():
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
    result = apply_log_transform_to_columns(result)
    result = add_transformed_columns(result)
    result = add_product_of_two_columns(result)
    return result


def apply_log_transform_to_columns(data):
    result: pd.DataFrame = data.copy()
    log = get_transformation_name_to_transformation_dictionary()['Log']
    column_names_for_log_transformation = get_column_names_for_log_transformation()
    for column_name in column_names_for_log_transformation:
        result[column_name] = result[column_name].apply(log)
    return result


def add_transformed_columns(data):
    result: pd.DataFrame = data.copy()
    transformation_names = get_transformation_names()
    transformation_dictionary = get_transformation_name_to_transformation_dictionary()
    column_name_list_dictionary = get_transformation_name_to_list_of_columns_dictionary()

    for transformation_name in transformation_names:
        column_names_for_transformation = column_name_list_dictionary[transformation_name]
        transformation = transformation_dictionary[transformation_name]
        for column_name in column_names_for_transformation:
            result[column_name + transformation_name] = result[column_name].apply(transformation)

    return result


def add_product_of_two_columns(data):
    result: pd.DataFrame = data
    pairs_of_column_names = get_pairs_of_column_names_for_adding_product()
    for pair_of_column_names in pairs_of_column_names:
        column_name_0 = pair_of_column_names[0]
        column_name_1 = pair_of_column_names[1]
        result[column_name_0 + 'Times' + column_name_1] = result[column_name_0] * result[column_name_1]
    return result


def get_transformation_names():
    return {'Log', 'Square', 'Root'}


def get_column_names_for_log_transformation():
    return ['LotArea', '1stFlrSF', '2ndFlrSF', 'GarageCars']


def get_pairs_of_column_names_for_adding_product():
    return [['OverallQual', 'OverallCond'], ['LotArea', 'OverallCond']]


def get_transformation_name_to_transformation_dictionary():
    log = lambda x: math.log2(x + 1)
    square = lambda x: x*x
    root = lambda x: math.sqrt(x)
    return {
        'Log': log,
        'Square': square,
        'Root': root
    }


def get_transformation_name_to_list_of_columns_dictionary():
    column_names_for_adding_log_transformation = ['OverallQual']
    column_names_for_adding_quadratic_transformation = ['LotArea', 'OverallQual', 'OverallCond', '1stFlrSF', '2ndFlrSF', 'GarageCars']
    column_names_for_adding_root_transformation =['2ndFlrSF', 'GarageCars']
    return {
        'Log': column_names_for_adding_log_transformation,
        'Square': column_names_for_adding_quadratic_transformation,
        'Root': column_names_for_adding_root_transformation,
    }


def difference_between_lists(original, removed):
    return list(set(original) - set(removed))
