# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 11:30:52 2016

@author: mathias
"""


#%%
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from battery_paper import BatteryPaper
from battery_pro import BatteryPro
from battery_regression import BatteryRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import cross_validation
import seaborn
from sklearn.tree import DecisionTreeClassifier

#%%
'''
Defining constants..
'''
path_dataset = '../preprocessed/preprocessed_user1.csv'

types = {'milliseconds': np.float64,# np.dtype(int),
         'index': np.int64,# np.dtype(int),
     'day': np.int64,# np.dtype(int),
     'hour': np.int64,# np.dtype(int),
     'battery_level' : np.int64,# np.dtype(int),
     'battery_plugged': np.int64,# np.dtype(int),
     'bright-level': np.int64,# np.dtype(int),
     'screen_on' : np.int64,# np.dtype(int),
     'conn': np.int64,# np.dtype(int),
     'conn_wifi': np.int64,# np.dtype(int),
     'temperature' : np.int64,# np.dtype(int),
     'voltage': np.int64,# np.dtype(int),
     'battery_level_grouped': np.int64,# np.dtype(int),
     'battery_common_smooth': np.int64,# np.dtype(int),
     'battery_exp_smooth': np.int64,# np.dtype(int),
     'no_day': np.int64, 
     'relative_day': np.int64, 
     'relative_min': np.int64,
     }

columns = [u'milliseconds', u'day', u'hour', u'battery_level', u'battery_plugged',
       u'bright_level', u'screen_on', u'conn', u'conn_wifi', u'temperature',
       u'voltage', u'battery_level_grouped', u'battery_common_smooth',
       u'battery_exp_smooth', u'no_day', u'relative_day', u'relative_min']
       
#%%
def mean_square_error(y_true, y_hat):
    error = np.subtract(y_true, y_hat)
    square_error = np.multiply(error, error)
    
    return round(np.sqrt(sum(np.divide(square_error, len(error)))), 2)
#%%
data_set = pd.read_csv(path_dataset, names=columns)

#%%
battery_regression = BatteryRegression()
battery_regression.fit(data_set)

#%%
initial_pos = 8039#int(np.random.rand()*100000)
steps = 600

fin = initial_pos + steps
mins = np.array(data_set['relative_min'][initial_pos:fin])
y_true = np.array(data_set['battery_exp_smooth'][initial_pos:fin])

regr_pred_simple, states_simple = battery_regression.predict_level(initial_pos=initial_pos,\
                                           steps=steps, other_models=False)

regr_pred, states = battery_regression.predict_level(initial_pos=initial_pos, steps=steps)
states['battery_exp_smooth'] = regr_pred
#%%
features = ["day", "hour", "relative_min", "battery_plugged", "screen_on", \
          "conn", "bat_prev_2", "bat_prev_1", "sin", "cos", "bat_prev_1_square", "prev1_plugged"]
pd.DataFrame(battery_regression.__regr__.coef_, index=features)

#%%
plt.plot(mins, y_true, 'bo',
         mins, regr_pred_simple, 'ro',
         mins, regr_pred, 'go')
plt.legend(('true', 'simple', 'others'))

#%%
def get_feature_model(features, target, model, data_set, offset=0):
    '''
    Given certain features, a target column and the model that will be used
    to predict the target, this function returns the model fitted.
    
    Before fitting the model, it splits the data in train and test data 
    sets, and it also takes into account an offset in case the target
    feature is also in the features set. That is done when the the 
    prediction is based on the previous state.
    '''
    end = data_set.shape[0]
    X = pd.DataFrame(data_set[features][:end-offset], columns=features, dtype=float)
    y = np.asarray(data_set[target][offset:], dtype=int)
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, \
                                                test_size=0.3, random_state=0)
    
    model.fit(X_train, y_train)
    print(model)
    
    return model
    
#%%

features = ["day", "hour", "bat_prev_1", "battery_plugged"]
target = "battery_plugged"
#plugged_model = get_feature_model(features, target, DecisionTreeClassifier(),\
#                                          data_set, 1)

#%%

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
#from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import make_pipeline
'''
n_estimator = 10
X = pd.DataFrame(data_set[features][:end-1], columns=features, dtype=float)
y = np.array(data_set[target][1:], dtype=int)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)
# It is important to train the ensemble of trees on a different subset
# of the training data than the linear regression model to avoid
# overfitting, in particular if the total number of leaves is
# similar to the number of training samples
X_train, X_train_lr, y_train, y_train_lr = cross_validation.train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.5)

# Unsupervised transformation based on totally random trees
rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
    random_state=0)

rt_lm = LogisticRegression()
pipeline = make_pipeline(rt, rt_lm)
pipeline.fit(X_train, y_train)
y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt, pos_label=1)

# Supervised transformation based on random forests
rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegression()
rf.fit(X_train, y_train)
rf_enc.fit(rf.apply(X_train))
rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm, pos_label=1)

grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression()
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)

y_pred_grd_lm = grd_lm.predict_proba(
    grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm, pos_label=1)


# The gradient boosted model by itself
y_pred_grd = grd.predict_proba(X_test)[:, 1]
fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd, pos_label=1)


# The random forest model by itself
y_pred_rf = rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf, pos_label=1)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
plt.plot(fpr_grd, tpr_grd, label='GBT')
plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
plt.plot(fpr_grd, tpr_grd, label='GBT')
plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()
'''

#%%


def predict_level(model, X, initial_pos=1, steps = 30):
    '''
    Once the model is fitted, this function returns the prediction for the
    minutes that are within the range initial_pos and initial_pos + steps.
    
    - initial_pos: initial position from which to start the prediction.
    - steps: number of rows to go ahead while doing the prediction.
    - other_models: this boolean variable indicates whether or not the 
    collaborating models should be used while carrying out the predictions.
    '''
    
    # We get the initial state
    initial_state = X[initial_pos:initial_pos+1]
    
    # Initial states both, for X and y
    state = initial_state.copy()
    predictions = [int(initial_state['battery_plugged'])]
    states = pd.DataFrame(columns=state.columns)
    states = states.append(state, ignore_index=True)
    
    for i in range(1, steps):
        # We make a prediction, within the range [0:100]
        prediction = model.predict_proba(state)[:, 1]
        predictions = predictions + [prediction]
        
        pos = i+initial_pos
        state['bat_prev_1'] = float(X['bat_prev_1'][pos:pos+1])
        state['hour'] = int(X['hour'][pos:pos+1])#-= 1440
        state['day'] = int(X['day'][pos:pos+1])#+=+ 1
        state["battery_plugged"] = 1 if prediction>0.5 else 0
    
    return predictions

#%%
features = ["day", "hour", "bat_prev_1", "battery_plugged"]
steps = 1000
for i in range(3):
    init = int(np.random.rand()*100000)
    print "Init position:", init
    end = init + steps
    
    n_estimator = 10
    X = pd.DataFrame(data_set[features][:-1], columns=features, dtype=float)
    y = np.array(data_set[target][1:], dtype=int)
    
    X_test = X[init:end]
    y_true = y[init:end]
    X_train, _, y_train, _ = cross_validation.train_test_split(X, y, test_size=0.5)
    # It is important to train the ensemble of trees on a different subset
    # of the training data than the linear regression model to avoid
    # overfitting, in particular if the total number of leaves is
    # similar to the number of training samples
    X_train, X_train_lr, y_train, y_train_lr = cross_validation.train_test_split(X_train,
                                                                y_train,
                                                                test_size=0.5)
    
    # Unsupervised transformation based on totally random trees
    rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
        random_state=0)
    
    rt_lm = LogisticRegression()
    pipeline = make_pipeline(rt, rt_lm)
    pipeline.fit(X_train, y_train)
    #y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
    y_pred_rt = predict_level(pipeline, X, init, steps)
    fpr_rt_lm, tpr_rt_lm, rt_lm_thresh = roc_curve(y_true, y_pred_rt, pos_label=1)
    print(auc(fpr_rt_lm, tpr_rt_lm))
    print(rt_lm_thresh)
    
    # Supervised transformation based on random forests
    rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
    rf_enc = OneHotEncoder()
    rf_lm = LogisticRegression()
    rf.fit(X_train, y_train)
    rf_enc.fit(rf.apply(X_train))
    rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
    '''
    #y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
    y_pred_rf_lm = predict_level(rf_lm, rf_enc.transform(rf.apply(X_test)), init, steps)
    fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm, pos_label=1)
    '''
    grd = GradientBoostingClassifier(n_estimators=n_estimator)
    grd_enc = OneHotEncoder()
    grd_lm = LogisticRegression()
    grd.fit(X_train, y_train)
    grd_enc.fit(grd.apply(X_train)[:, :, 0])
    grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
    '''
    y_pred_grd_lm = grd_lm.predict_proba(
        grd_enc.transform(grd.apply(X)[:, :, 0]))[:, 1]
    fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_true, y_pred_grd_lm, pos_label=1)
    
    '''
    # The gradient boosted model by itself
    #y_pred_grd = grd.predict_proba(X_test)[:, 1]
    y_pred_grd = predict_level(grd, X, init, steps)
    fpr_grd, tpr_grd, grd_thresh = roc_curve(y_true, y_pred_grd, pos_label=1)
    print(auc(fpr_grd, tpr_grd))
    print(grd_thresh)
    # The random forest model by itself
    #y_pred_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = predict_level(rf, X, init, steps)
    fpr_rf, tpr_rf, rf_thresh = roc_curve(y_true, y_pred_rf, pos_label=1)
    print(auc(fpr_rf, tpr_rf))
    print(rf_thresh)
    
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    #plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
    plt.plot(fpr_grd, tpr_grd, label='GBT')
    #plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


