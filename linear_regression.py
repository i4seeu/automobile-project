#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 08:27:53 2019

@author: software-engineer
"""

import pandas as pd
import numpy as np


data = pd.read_csv('Automobile_data.csv')
# Preprocess the dataset by coercing the important columns to numeric values
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')
data['price'] = pd.to_numeric(data['price'], errors='coerce')

# And removing any rows which contain missing data
data.dropna(subset=['price', 'horsepower'], inplace=True)

from scipy.stats.stats import pearsonr
pearsonr(data['horsepower'], data['price'])

#data visualization
from bokeh.io import output_notebook
from bokeh.plotting import ColumnDataSource, figure, show

# enable notebook output
output_notebook()

source = ColumnDataSource(data=dict(
    x=data['horsepower'],
    y=data['price'],
    make=data['make'],
))

tooltips = [
    ('make', '@make'),
    ('horsepower', '$x'),
    ('price', '$y{$0}')
]

p = figure(plot_width=600, plot_height=400, tooltips=tooltips)
p.xaxis.axis_label = 'Horsepower'
p.yaxis.axis_label = 'Price'

# add a square renderer with a size, color, and alpha
p.circle('x', 'y', source=source, size=8, color='blue', alpha=0.5)

# show the results
p.show()

# split our data into train (75%) and test (25%) sets
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.25)

from sklearn import linear_model
model = linear_model.LinearRegression()
# the linear regression model expects a 2d array, so we add an extra dimension with reshape
# input: [1, 2, 3], output: [ [1], [2], [3] ]
# this allows us to regress on multiple independent variables later
training_x = np.array(train['horsepower']).reshape(-1, 1)
training_y = np.array(train['price'])
# perform linear regression
model.fit(training_x, training_y)
# output is a nested array in the form of [ [1] ]
# squeeze removes all zero dimensions -> [1]
# asscalar turns a single number array into a number -> 1
slope = np.asscalar(np.squeeze(model.coef_))
intercept = model.intercept_
print('slope:', slope, 'intercept:', intercept)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# define a function to generate a prediction and then compute the desired metrics
def predict_metrics(lr, x, y):
    pred = lr.predict(x)
    mae = mean_absolute_error(y, pred)
    mse = mean_squared_error(y, pred)
    r2 = r2_score(y, pred)
    return mae, mse, r2

training_mae, training_mse, training_r2 = predict_metrics(model, training_x, training_y)

# calculate with test data so we can compare
test_x = np.array(test['horsepower']).reshape(-1, 1)
test_y = np.array(test['price'])
test_mae, test_mse, test_r2 = predict_metrics(model, test_x, test_y)

print('training mean error:', training_mae, 'training mse:', training_mse, 'training r2:', training_r2)
print('test mean error:', test_mae, 'test mse:', test_mse, 'test r2:', test_r2)