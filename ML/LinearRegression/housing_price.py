#!/usr/bin/python 
#-*- coding:utf8 -*-

from __future__ import print_function
import tensorflow as tf

import os
import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plot 


def maxminnorm(arr):
    maxcols = arr.max(axis=0)
    mincols = arr.min(axis=0)
    data_shape = arr.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    res = np.empty((data_rows, data_cols))
    for i in xrange(data_cols):
        res[:, i] = (arr[:, i] - mincols[i])/(maxcols[i]-mincols[i]) * 10000
    return res



housing = pd.read_csv('./kc_train_data.csv')
target = pd.read_csv('./kc_target.csv')

tr_housing_dat = np.delete(housing.values,1, axis=1)
tr_target_dat = target.values

tr_housing_dat = maxminnorm(tr_housing_dat)

samples = tr_housing_dat.shape[0]
columns = tr_housing_dat.shape[1]
weights = tf.Variable(initial_value=tf.random.normal(shape=(columns, 1)))
bias = tf.Variable(initial_value=tf.random.normal(shape=(1, 1)))

x = tf.placeholder(tf.float32, [None, columns])
tr_pred = tf.matmul(x, weights) + bias

error = tf.reduce_mean(tf.square(tr_pred-tr_target_dat))

#optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(error)
optimizer = tf.train.AdamOptimizer(0.01).minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for t in tr_housing_dat:
        _, e, w_value, b_value = sess.run([optimizer, error, weights, bias ], feed_dict={x:[t]})


dat=np.dot(tr_housing_dat, w_value[:, 0]) + b_value[0]
print(dat)
print(tr_target_dat)


plot.figure(figsize=(10, 7))
num = 100
x = np.arange(1, num + 1)
plot.plot(x, tr_target_dat[:num], label = 'target')
plot.plot(x, dat[: num], label='preds')
plot.legend(loc='upper right')
plot.show()




