# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 12:04:49 2016

@author: mathias
"""

import numpy as np


class BatteryPaper:
    
    def __init__(self):
        self.__is_fitted = False

    def fit(self, data_set):
        '''
        Given the data set with all the information, this function fits the 
        data for future predictions.
        
        - data_set: pandas Dataframe which contains all the necessary information.
        At least it must have the following features: day, hour, relative_min,
        battery-plugged, screen-on, conn, temperature, and battery_exp_smooth.
        '''
        self.__data_set = data_set
        data_set['use_time'] = np.append(np.subtract(data_set['relative_min'][1:], data_set['relative_min'][:-1]), [5])
        sums = data_set[['screen_on', 'conn', 'conn_wifi', 'use_time', 'battery_exp_smooth']].\
                                       groupby(['screen_on', 'conn', 'conn_wifi']).sum()
        total_time = data_set['use_time'].sum()
        total_battery = data_set['battery_exp_smooth'].sum()
        sums['use_time'] = sums['use_time'] / total_time
        sums['battery_exp_smooth'] = sums['battery_exp_smooth'] / total_battery

        self.__average = np.dot(sums['use_time'], sums['battery_exp_smooth'])
        self.__is_fitted = True


    def predict_level(self, mins, initial_pos):
        '''
        Given a list of minutes with an offset from the initial point used to
        fit the model, it will return a list of battery levels that stand for
        the prediction of the model for those minutes.
        - initial_pos: position from where the model will make the predictions.
        '''
        self.__actual_percentage = int(self.__data_set['battery_exp_smooth'][initial_pos:initial_pos+1])
        self.__now = int(self.__data_set['relative_min'][initial_pos:initial_pos+1])
        
        assert self.__is_fitted, "The model is not fitted yet to make a prediction."
        if type(mins) != np.ndarray:
            mins = np.array(mins)
        
        predictions = []
        for x in mins:
            prediction = (x - self.__now) * self.__average + self.__actual_percentage
            if prediction > 100:
                prediction = 100
            if prediction < 0:
                prediction = 0
            predictions = predictions + [prediction]
        
        return(predictions)
        



