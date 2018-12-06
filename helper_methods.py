import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
from random import uniform, gauss


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
    trainX = sm.add_constant(data=trainX, has_constant='add')
    trainY = training_set['SalePrice']
    testX = testing_set.drop(columns=['SalePrice'])
    testX = sm.add_constant(data=testX, has_constant='add')
    testY = testing_set['SalePrice']
    return [trainX, trainY, testX, testY]

def naive_transformation(data):
    return data


def label_categorical_features(data_set, column_names):
    return pd.get_dummies(data=data_set, prefix=column_names)


def root_mean_square_error(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise Exception('Vectors do not have the same type')
    diff = y_pred - y_true
    return math.sqrt(diff.T.dot(diff)/len(diff))

