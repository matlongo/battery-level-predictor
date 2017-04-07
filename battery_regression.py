# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 09:41:52 2016

@author: mathias
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize


class BatteryRegression:
    '''
    This class represents the regression model used to predict the level of
    battery for the next state. Each state differs from the previous one in 
    5 minutes approximately.
    
    The prediction can be done solely using the initial state, or it can also
    consider other models that can serve to predict future states, such as
    changes in battery charger connected, connections to some network, screen
    on, and so forth.
    '''
    
    
    def __init__(self):
        self.__is_fitted = False
        self.__TRAIN_PGE = .1
        
    
    def fit(self, data_set):
        '''
        Give a data set, this function trains the regression model underlying
        this class. In order to do so, the function splits the data in train
        and test sets.
        
        Besides, it trains the models that serve to predict a state, i.e.,
        models to predict the temperature, and the state (on/off) of the screen,
        the charger, and the connection to some network.
        
        - data_set: pandas Dataframe which contains all the necessary information.
        At least it must have the following features: day, hour, relative_min,
        battery_plugged, screen_on, conn, temperature, and battery_exp_smooth.
        '''
        self.__data_set = data_set

        mins_per_day = 60 * 24
        self.__battery_level_mean = self.__data_set['battery_exp_smooth'].mean()
        self.__battery_plugged_mean = self.__data_set['battery_plugged'].mean()
        amplitude = min(100 - self.__battery_level_mean, self.__battery_level_mean - 0)
        amplitude_plugged = min(1 - self.__battery_plugged_mean, self.__battery_plugged_mean - 0)
        
        self.__data_set['sin'] = np.sin(np.pi * 2 * self.__data_set['hour'] / mins_per_day) * amplitude + self.__battery_level_mean
        self.__data_set['cos'] = np.cos(np.pi * 2 * self.__data_set['hour'] / mins_per_day) * amplitude + self.__battery_level_mean
        self.__data_set['cos_plugged'] = np.cos(np.pi * 2 * self.__data_set['hour'] / mins_per_day) * amplitude_plugged + self.__battery_plugged_mean
        first_bat = self.__data_set['battery_exp_smooth'].iloc[0]
        self.__data_set['bat_prev_1'] = np.append([first_bat], self.__data_set['battery_exp_smooth'][:-1])
        self.__data_set['bat_prev_2'] = np.append([first_bat, first_bat], self.__data_set['battery_exp_smooth'][:-2])
        self.__data_set['bat_prev_1_square'] = np.square(self.__data_set['bat_prev_1'])
        self.__data_set['prev1_plugged'] = np.multiply(self.__data_set['bat_prev_1'], self.__data_set['battery_plugged'])
        
        # Features used to fit the regression model
        features = ["day", "hour", "relative_min", "battery_plugged", "screen_on", \
            "conn", "bat_prev_2", "bat_prev_1", "sin", "cos", "bat_prev_1_square", "prev1_plugged", "cos_plugged"]# "temperature",
        target = "battery_exp_smooth"
        
        self.__X = self.__data_set[features][:-1]
        self.__y = np.asarray(self.__data_set[target], dtype=int)[1:]
        
        # We split the data before fitting the model
        train_len = int(self.__X.shape[0] * self.__TRAIN_PGE)
        X_train = self.__X[:train_len]
        y_train = self.__y[:train_len]
        #X_test = self.__X[150000:200000]
        #y_test = self.__y[150000:200000]
        #X_train, X_test, y_train, y_test = cross_validation.\
        #      train_test_split(self.__X, self.__y, test_size=0.3, random_state=0)
        '''
        # We sort the data before fitting the model
        X_train['target'] = y_train
        X_train = X_train.sort("relative_min")
        y_train = np.array(X_train['target'])
        X_train = X_train.drop('target', 1)
        X_test['target'] = y_test
        X_test = X_test.sort("relative_min")
        y_test = np.array(X_test['target'])
        X_test = X_test.drop('target', 1)
        '''
        
        data, self.__norms = normalize(X_train, return_norm=True, axis=0)
        self.__norms = pd.Series(self.__norms, index=features)
        # Create linear regression object
#        self.__regr__ = linear_model.LinearRegression()
        self.__regr__ = linear_model.Lasso(alpha=0.0001)
        # Train the model using the training sets
        self.__regr__.fit(data, y_train)
        # Train the models to predict the other features
        self.__fit_other_models()
        # Once everything went fine, we can assume that the model is fitted
        self.__is_fitted = True
        
        
    def __fit_other_models(self):
        '''
        This private function aims at fitting the collaborating models.
        '''
        self.__other_models = dict()
        
        features = ["day", "hour", "screen_on", "temperature"]
        target = "temperature"
        temp_model = self.__get_feature_model(features, target, DecisionTreeRegressor(max_depth=6),\
                                               self.__data_set, 1)
        #self.__other_models['temperature'] = (temp_model, features[:])
        
        features = ["day", "hour", "screen_on"]
        target = "screen_on"
        screen_model = self.__get_feature_model(features, target, DecisionTreeClassifier(),\
                                                  self.__data_set, 1)
        self.__other_models["screen_on"] = (screen_model, features[:], 0.5)
        
        features = ["day", "hour", "conn"]
        target = "conn"
        conn_model = self.__get_feature_model(features, target, DecisionTreeClassifier(),\
                                                  self.__data_set, 1)
        self.__other_models['conn'] = (conn_model, features[:], 0.5)

        features = ["day", "hour", "cos_plugged", "bat_prev_1", "battery_plugged"]
        target = "battery_plugged"
        plugged_model = self.__get_feature_model(features, target, RandomForestClassifier(max_depth=3, n_estimators=10),\
                                                  self.__data_set, 1)
        
        self.__other_models['battery_plugged'] = (plugged_model, features[:], 0.94)
    
    
    
    def __get_feature_model(self, features, target, model, data_set, offset=0):
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
        
        return model
        
        
    
    def predict_level(self, initial_pos=1, steps = 30, other_models = True):
        '''
        Once the model is fitted, this function returns the prediction for the
        minutes that are within the range initial_pos and initial_pos + steps.
        
        - initial_pos: initial position from which to start the prediction.
        - steps: number of rows to go ahead while doing the prediction.
        - other_models: this boolean variable indicates whether or not the 
        collaborating models should be used while carrying out the predictions.
        '''
        assert self.__is_fitted, "First you have to fit the model."
        assert 0 < initial_pos and initial_pos < self.__X.shape[0], "The position given\
                                          is out of scope."
        assert steps > 0, "The number of steps must be greater than 0"
        #assert initial_pos + steps < self.__X.shape[0], "The number of steps plus\
        #                              the initial_pos is out of scope." 
        
        # We get the initial state
        initial_state = self.__X[initial_pos:initial_pos+1]
        
        # Initial states both, for X and y
        state = initial_state.copy()
        predictions = [float(initial_state['bat_prev_1'])]
        states = pd.DataFrame(columns=state.columns)
        states = states.append(state, ignore_index=True)
        
        for i in range(1, steps):
            # We make a prediction, within the range [0:100]
            state_to_predict = pd.Series(index=state.columns)
            for column in state.columns:
                state_to_predict[column] = float(state[column]) / self.__norms[column]
            prediction = max(min(float(self.__regr__.predict(state_to_predict.reshape(1, -1))[0]), 100), 0)
            predictions = predictions + [prediction]
            
            # We update the state for following predictions
            if other_models:
                for key, value in self.__other_models.iteritems():
                    state[key] = 1 if value[0].predict(state[value[1]])[0] > value[2] else 0
            state['bat_prev_2'] = state['bat_prev_1']
            state['bat_prev_1'] = prediction
            
            #pos = i+initial_pos
            state['relative_min'] += 1
            state['hour'] = state['hour'] + 1
            if (int(state['hour']) >= 1440):
                state['hour'] = 0
                state['day'] = (state['day'] + 1) % 7
            #state['day'] = int(self.__X['day'][pos:pos+1])#+=+ 1
            amplitude = min(100 - self.__battery_level_mean, self.__battery_level_mean - 0)
            amplitude_plugged = min(1 - self.__battery_plugged_mean, self.__battery_plugged_mean - 0)
            state['sin'] = np.sin(np.pi * state['hour'] / 1440) * amplitude + self.__battery_level_mean
            state['cos'] = np.cos(np.pi * state['hour'] / 1440) * amplitude_plugged + self.__battery_plugged_mean
            state['cos_plugged'] = np.cos(np.pi * state['hour'] / 1440) * amplitude + self.__battery_level_mean
            state['prev1_plugged'] = state['bat_prev_1'] * state['battery_plugged']
            state['bat_prev_1_square'] = np.square(state['bat_prev_1'])
            
            states = states.append(state, ignore_index=True)
        
        return (predictions, states)
        
        
        
        
        
        
        