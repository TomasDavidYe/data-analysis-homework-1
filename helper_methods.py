import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
from random import uniform, gauss


def label_categorical_features(data_set, column_names):
    return pd.get_dummies(data=data_set, prefix=column_names)


def mean_square_error(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise Exception('Vectors do not have the same type')
    diff = y_pred - y_true
    return math.sqrt(diff.T.dot(diff)/len(diff))


def show_scatter_graph(data, x, y):
    plt.scatter(x=data[x], y=data[y])
    plt.show()


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

    # Part 2: Training a naive model
    evaluation_data.drop(columns=['Id'], inplace=True)
    development_data.drop(columns=['Id'], inplace=True)
    return [development_data, evaluation_data]





def show_box_plot(data):
    plt.boxplot(data=data)
