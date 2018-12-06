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
    return diff.T.dot(diff)/len(diff)


def show_scatter_graph(data, x, y):
    plt.scatter(x=data[x], y=data[y])
    plt.show()


def show_box_plot(data):
    plt.boxplot(data=data)
