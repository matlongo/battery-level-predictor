# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:50:49 2016

@author: mathias
"""

'''
Importing the libraries..
'''
import numpy as np


class BatteryPro:
    '''
    This class aims at representing the battery pro application model, used to
    predict the battery level in a particular time.
    '''
        
    def __init__(self):
        self.__isFitted = False
    
    def get_average_time_by_time(self, mins):
        '''
        This function returns the average time it takes to the mobile phone to
        change one point of battery level, either upside or downside, based on
        the consumption of the last minutes given by parameters.
        
        - mins: the number of minutes to take into account to carry out the 
        average.
        '''
        levels = np.array(self.__data_set['battery_exp_smooth'])
        i = mins - 1
        battery_change = 0.
        
        while i > 0:# and  now - mins < minutes[i]:
            i -= 1
            change = levels[i+1] - float(levels[i])
            # If it was decreasing and now it started to increase, the rest of
            # the information will be worthless. The same for the viceversa case.
            if change * battery_change < 0:
                break
            battery_change += change
                
        avg_time = battery_change / mins
        return(avg_time)
       
    
    def get_average_time_by_battery(self, points=3, initial_pos=0):
        '''
        This function returns the average time it takes to the mobile phone to
        change one point of battery level, either upside or downside, based on
        the consumption of the last battery point changes.
        
        - points: the number of points to take into account to carry out the 
        average.
        - initial_pos: the position in the dataset where to start carrying out
        the average.
        '''
        minutes = np.array(self.__data_set['relative_min'])
        levels = np.array(self.__data_set['battery_exp_smooth'])
        i = initial_pos-2
        
        while i < self.__data_set.shape[0]  and \
              np.abs(self.__actual_percentage - levels[i]) < points:
            i += 1
        
        total_battery_change = np.abs(levels[i] - self.__actual_percentage)
        total_minutes_change = np.abs(minutes[i] - self.__now)
        
        avg_time = float(total_battery_change) / total_minutes_change
        
        return avg_time
       
       
    def fit_by_time(self, data_set, mins=5):
        '''
        This function initializes all the variables necessary to make a later 
        estimation. After this method is called, the object is ready to make
        predictions.
        '''
        assert mins < data_set.shape[0], "Mins must be less than the size of the dataset."
        assert mins >= 1, "Mins must be greater than or equal to one."
        
        self.__by_time = True#
        self.__data_set = data_set
        self.__mins = mins
        self.__isFitted = True

        
    def fit_by_battery(self, data_set, points=5):
        '''
        This function initializes all the variables necessary to make a later 
        estimation. After this method is called, the object is ready to make
        predictions.
        '''
        assert points < self.__data_set.shape[0], "Mins must be less than the size of the dataset."
        assert points >= 1, "Mins must be greater than or equal to one."
        
        self.__by_time = False#self.get_average_time_by_battery(points, initial_pos)
        self.__data_set = data_set
        self.__points = points
        self.__isFitted = True
                
        
    def predict_level(self, mins, initial_pos=0):
        '''
        Given a list of minutes with an offset from the initial point used to
        fit the model, it will return a list of battery levels that stand for
        the prediction of the model for those minutes.
        '''
        assert self.__isFitted, "The model is not fitted yet to make a prediction."
        assert initial_pos >= 0, "The initial_pos must be greater than or equal to zero."
        
        if type(mins) != np.ndarray:
            mins = np.array(mins)
            
        self.__actual_percentage = int(self.__data_set['battery_exp_smooth'][initial_pos:initial_pos+1])
        self.__now = int(self.__data_set['relative_min'][initial_pos:initial_pos+1])
        
        
        if (self.__by_time):
            self.__avg_time = self.get_average_time_by_time(self.__mins)
        else:
            self.get_average_time_by_battery(self.__points, initial_pos)
        
        predictions = []
        for x in mins:
            prediction = (x - self.__now) * self.__avg_time + self.__actual_percentage
            if prediction > 100:
                prediction = 100
            if prediction < 0:
                prediction = 0
            predictions = predictions + [prediction]
        
        return(predictions)
            
    
   

