# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 16:20:46 2016

@author: mathias
"""
#%%
'''
Importing the libraries..
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeRegressor
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
import seaborn
from battery_pro import BatteryPro
from battery_regression import BatteryRegression
from battery_paper import BatteryPaper

#%%
'''
Defining constants..
'''
path_dataset = 'preprocessed/preprocessed_user1.csv'

types = {'milliseconds': np.float64,# np.dtype(int),
         'index': np.int64,# np.dtype(int),
     'day': np.int64,# np.dtype(int),
     'hour': np.int64,# np.dtype(int),
     'battery-level' : np.int64,# np.dtype(int),
     'battery-plugged': np.int64,# np.dtype(int),
     'bright-level': np.int64,# np.dtype(int),
     'screen-on' : np.int64,# np.dtype(int),
     'conn': np.int64,# np.dtype(int),
     'conn-wifi': np.int64,# np.dtype(int),
     'temperature' : np.int64,# np.dtype(int),
     'voltage': np.int64,# np.dtype(int),
     'battery-level-grouped': np.int64,# np.dtype(int),
     'battery-common-smooth': np.int64,# np.dtype(int),
     'battery-exp-smooth': np.int64,# np.dtype(int),
     }

columns = ["milliseconds", "day", "hour", "battery-level", "battery-plugged", \
            "bright-level", "screen-on", "conn", "conn-wifi", "temperature", \
            "voltage", "battery-level-grouped", 'battery-common-smooth', \
            'battery-exp-smooth']

#%%
'''
Loading the dataset..
'''
data_set = pd.read_csv(path_dataset, names=columns, dtype=types)
days = np.array(data_set['day'])
        
days_range = map(lambda x: x+7 if x<0 else x, np.subtract(days[1:], days[:-1]))
data_set['no_day'] = np.cumsum(np.append([0], days_range))
minutes_per_day = 60 * 24
data_set['relative_min'] = map(lambda day, minu: day * minutes_per_day+ minu,\
                            data_set['no_day'], data_set['hour'])

#%%
'''
In this block we provide two functions. Both of them return a regression model
given the dataset.
'''
                                            
def simple_regression(X, y, X_test, y_test):
    '''
    Given X and y, this function returns and evaluates a simple linear 
    regression model.
    '''    
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(X, y)
    
    residual = np.mean((regr.predict(X_test) - y_test) ** 2)
    score = regr.score(X_test, y_test)
    
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean square error
    print("Residual sum of squares: %.2f" % residual)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % score)
    print('Variance score on training set: %.2f' % regr.score(X, y))
    
    return (regr, score, residual)
   
   
def tree_regression(X, y, X_test, y_test):
    '''
    Given X and y, this function returns and evaluates a decision tree
    regression model.
    '''    
    regr_2 = DecisionTreeRegressor(max_depth=5)

    regr_2.fit(X, y)
    
    residual = np.mean((regr_2.predict(X_test) - y_test) ** 2)
    score = regr_2.score(X_test, y_test)
    
    # The mean square error
    print("Residual sum of squares: %.2f" % residual)
    # Explained variance score: 1 is perfect prediction
    print('Variance score on test set: %.2f' % score)
    print('Variance score on training set: %.2f' % regr_2.score(X, y))
    
    return (regr_2, score, residual)

def logistic_regression_model(X, y, X_test, y_test):
    OVO = OneVsOneClassifier(LogisticRegression()).fit(X,y)
    print "One vs one accuracy: %.3f" % OVO.score(X_test,y_test)
    return OVO
    

def fill_min_gaps(data_set, columns):
    result = pd.DataFrame(columns=columns)
    last_row = next(data_set.iterrows())[1]
    result = result.append(last_row)
    
    for index, row in data_set[1:].iterrows():
        diff_mins = row['relative_min'] - last_row['relative_min']
        diff_batt = row['battery-exp-smooth'] - last_row['battery-exp-smooth']
        average_cons = diff_batt / float(diff_mins)    
        
        new_row = last_row[:]
        for i in range(1, int(diff_mins)):
            new_row['relative_min'] += 1
            new_row['battery-exp-smooth'] += average_cons
            #print new_row, type(new_row)
            result = result.append(new_row)
        
        result = result.append(row)
        last_row = row
    
    return result
    
#%%
'''
In this block we split the dataset in order to estimate the battery-level. For
this, we show and compare two different models: simple regression and decision
tree regressor.
'''
features = ["day", "hour", "relative_min", "battery-plugged", "screen-on", \
            "conn", "temperature", "battery-exp-smooth"]
target = "battery-exp-smooth"

X = pd.DataFrame(data_set[features][:-1], columns=features, dtype=int)
y = np.asarray(data_set[target][1:], dtype=int)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, \
                                            test_size=0.3, random_state=0)

X_train['target'] = y_train
X_train = X_train.sort("relative_min")
y_train = np.array(X_train['target'])
X_train = X_train.drop('target', 1)
X_test['target'] = y_test
X_test = X_test.sort("relative_min")
y_test = np.array(X_test['target'])
X_test = X_test.drop('target', 1)

(linear_reg_model, lin_score, lin_res) = simple_regression(X_train, y_train, \
                                                       X_test, y_test)
(tree_reg_model, tree_score, tree_res) = tree_regression(X_train, y_train, \
                                                         X_test, y_test)

#%%
'''
Plotting the residual sum of squares and the efficiency of the two regression
models.

labels = ('Linear Regression', 'Decision Tree Regressor')
indexes = np.array([.25, 1.])
efficiencies = [lin_score, tree_score]
residuals = [lin_res, tree_res]
width = .35

#fig = plt.figure()

#Efficiency
plt.bar(indexes, efficiencies, width, error_kw=dict(tick_label=labels))
plt.title('Efficiency')
plt.xticks(indexes+width/2., labels)
plt.ylim(.90,1.)
plt.show()

#Residual Sum of Squares
plt.bar(indexes, residuals, width, error_kw=dict(tick_label=labels))
plt.title('Residuals')
plt.xticks(indexes+width/2., labels)
plt.show()
'''

#%%
'''
We calculate the pvalues in this block. This is useful to know
which variables are really important. The variables below 0.05
are considered to be significant for the model.

scores, pvalues = chi2(X, y)
print pvalues
'''

#%%
'''
This block plots both the predicted points and the dataset values.

line2, = plt.plot(y_test[:100])
line, = plt.plot(linear_reg_model.predict(X_test[:100]))
plt.legend( (line, line2),
           ('Real value', 'Prediction'),
           'upper left' )
plt.show()
'''

#%%
'''
Here we try to use a Decision Tree Classifier, using the 'battery-level-grouped'
instead of the 'battery-level'.

features = ["day", "hour", "battery-plugged", "screen-on", \
            "conn", "battery-level-grouped"]#, "temperature"]
target = "battery-level-grouped"

X = pd.DataFrame(data_set[features][:-1], columns=features, dtype=float)
y = np.asarray(data_set[target][1:], dtype=int)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, \
                                            test_size=0.3, random_state=0)
                                            
log = DecisionTreeClassifier(splitter="random")
log.fit(X_train,y_train)
print log.score(X_test, y_test)
'''


#%%
'''
In this block we build up a model to estimate each of the features needed to
calculate the battery level.
'''
def get_feature_model(features, target, model, data_set, offset=0):
    end = data_set.shape[0]
    X = pd.DataFrame(data_set[features][:end-offset], columns=features, dtype=float)
    y = np.asarray(data_set[target][offset:], dtype=int)
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, \
                                                test_size=0.3, random_state=0)
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print 'Score: ', score
    
    return (model, score)

features = ["day", "hour", "screen-on", "temperature"]
target = "temperature"
(temp_model, temp_score) = get_feature_model(features, target, \
                                DecisionTreeRegressor(max_depth=6), data_set, 1)

features = ["day", "hour", "screen-on"]
target = "screen-on"
(screen_model, screen_score) = get_feature_model(features, target, \
                                DecisionTreeClassifier(), data_set, 1)

features = ["day", "hour", "conn"]
target = "conn"
(conn_model, conn_score) = get_feature_model(features, target, \
                                DecisionTreeClassifier(), data_set, 1)

features = ["day", "hour", "battery-exp-smooth", "battery-plugged"]
target = "battery-plugged"
(plugged_model, plugged_score) = get_feature_model(features, target, \
                                DecisionTreeClassifier(), data_set, 1)


#%%
'''
In this block we compare the error obtained from the two models previously
trained. We start by predicting the first step, and all the next predictions
are based on what has been previously calculated.
'''
def get_estimation(model, initial_state, real_values, steps=30, other_models=None):
    sq_error = 0
    accumulative_error = [0]
    state = initial_state.copy()
    states = pd.DataFrame(initial_state.copy())
    predictions = [float(initial_state['battery-exp-smooth'])]
    
    for i in range(steps):
        prediction = float(model.predict(state)[0])
        predictions = predictions + [prediction if prediction <= 100. else 100.]
        error = prediction - real_values[i:i+1][0]
        sq_error += np.sqrt(error * error)
        accumulative_error.append(sq_error)
        if other_models:
            for key, value in other_models.iteritems():
                state[key] = value[0].predict(state[value[1]])[0]
        state['battery-exp-smooth'] = prediction if prediction <= 100. else 100.
        state['battery-exp-smooth'] = prediction if prediction >= 0. else 0.
        state['hour'] += 5
        if (int(state['hour']) > 1440):
            state['hour'] -= 1440
            state['day'] +=+ 1
        
        states = states.append(state)

    return (predictions, accumulative_error, states)

steps = 200
ini = 1
fin = ini + steps
other_models = dict()
other_models['battery-plugged'] = (plugged_model, ["day", "hour", \
                                     "battery-exp-smooth", 'battery-plugged'])
other_models['conn'] = (conn_model, ["day", "hour", "conn"])
other_models['temperature'] = (temp_model, ["day", "hour", \
                                    "screen-on", "temperature"])
                                     
(predictions_linear, errors, states) = get_estimation(linear_reg_model, \
                     X[ini:ini+1], y[ini:fin], steps, other_models)
(predictions_linear_simple, errors2, states_linear_simple) = \
        get_estimation(linear_reg_model, X[ini:ini+1], y[ini:fin], steps)

x_range = range(steps+1)
line2, = plt.plot(x_range, errors2)
line, = plt.plot(x_range, errors)
plt.legend( (line, line2),
           ('Linear Regression', 'Linear Simple'),
           'upper left' )
plt.show()

#%%
new_data_set = fill_min_gaps(data_set[ini:ini+500], columns)
#%%
battery_pro = BatteryPro(new_data_set)
battery_pro.fit_by_battery(points=10, initial_pos=150)
#initial_time = X['relative_min'][ini:ini+1]
estimations = battery_pro.predict_level(np.array(X['relative_min'][ini:fin]))#.get_estimations(mins=10)

#%%
batt_reg = BatteryRegression()
batt_reg.fit(data_set)
predictions_linear = batt_reg.predict_level(initial_pos = ini, steps=steps)

#%%
batt_pap = BatteryPaper()
batt_pap.fit(data_set, ini)
predictions_linear = batt_pap.predict_level(np.array(X['relative_min'][ini:fin]))

#%%
slope = y[ini+1] - y[ini]
basic_model_predictions = np.multiply(range(steps), slope) + y[ini]
basic_model_predictions[basic_model_predictions > 100] = 100
basic_model_predictions[basic_model_predictions < 0] = 0

print '\n\n\n'
print '------------------------------------------'
print '----- R2  Values -------------------------'
print '------------------------------------------'
print 'Linear Regression Model w/other Models:', r2_score(y[ini:fin], predictions_linear)
print 'Basic Model:', r2_score(y[ini:fin], basic_model_predictions)
print 'Linear Regression Basic', r2_score(y[ini-1:fin], predictions_linear_simple)
print '------------------------------------------'

now = int(X['relative_min'][ini:ini+1])

line, = plt.plot(X['relative_min'][ini-1:fin], y[ini-1:fin])
line2, = plt.plot(X['relative_min'][ini:fin], predictions_linear)
#line3, = plt.plot(range(now, now+steps), basic_model_predictions)
#line4, = plt.plot(X['relative_min'][ini-1:fin], predictions_linear_simple)
#line5, = plt.plot(X['relative_min'][ini:ini+200], estimations)
plt.legend( (line, line2),#, line3, line4, line5),
           ('Real value', 'Prediction Linear Regression'), \
           #'Predicion Basic Model', 'Linear Simple', 'Battery Pro'),
           'upper left' )
plt.show()