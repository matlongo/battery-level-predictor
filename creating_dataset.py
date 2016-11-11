# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 15:08:53 2016

@author: mathias
"""
import pandas as pd
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
#import seaborn

#%%
user = sys.argv[1]
path = "users/" + user + ".csv"
path_to_write = "preprocessed/preprocessed_"+ user + ".csv"
columns = ["id", "milliseconds", "timestamp", "feature", "value"]
CHUNK_SIZE = 10000000

battery_pattern = "power\|battery\|level|power\|battery\|plugged"
screen_pattern = "screen\|brightness\|level|screen\|power"
conn_pattern = "conn\|.*\|detailedstate"
temperature_pattern = "power\|battery\|temperature"
voltage_pattern = "power\|battery\|voltage"
patterns = battery_pattern+"|"+screen_pattern+"|"+conn_pattern+"|"\
            +temperature_pattern+"|"+voltage_pattern

#%%
'''
In this block we get all the interesting features: screen, battery and connection.
'''
def matches(feature):
    '''
    This function returns a boolean stating whether or not the feature matches
    with some of the patterns.
    '''
    match = re.match(patterns, feature)
    if match:
        return True
    return False
    

def get_day_of_week(timestamp):
    '''
    In this function we split the timestamp into date and time. Furthermore, 
    the date is converted in an integer which stands for the day of the week, 
    where Monday is represented by zero and Sunday by 6.
    '''
    my_date = datetime.date(int(timestamp[0:4]), int(timestamp[5:7]), \
                                                     int(timestamp[8:10]))

    return my_date.weekday()


def to_minutes_wo_day(hour):
    hours = int(hour[0:2])
    minutes = int(hour[3:5])
    minutes = hours * 60 + minutes
    
    return int(minutes)
    
def to_minutes(hour, day):
    hours = int(hour[0:2])
    minutes = int(hour[3:5])
    minutes = day * 60 * 24 + hours * 60 + minutes
    
    return minutes

def get_battery_category(level):
    if level > 95:
        return 9
    if level > 85:
        return 8
    if level > 75:
        return 7
    if level > 65:
        return 6
    if level > 55:
        return 5
    if level > 45:
        return 4
    if level > 35:
        return 3
    if level > 25:
        return 2
    if level > 15:
        return 1
    return 0


def fill_gaps(data_set, row, last_row):
    minutes_per_day = 24*60
    
    diff_mins = row["hour"] - last_row["hour"]
    if diff_mins < 0:
        diff_mins = diff_mins + minutes_per_day
    
    #We fill the gaps in mins with the average battery consumption
    diff_batt = row['battery_level'] - last_row['battery_level']
    if diff_mins > 0:
        average_cons = diff_batt / float(diff_mins)    
    
    new_row = last_row.copy()
    for i in range(1, int(diff_mins)):
        new_row['hour'] += 1
        if (new_row['hour'] > minutes_per_day):
            new_row['hour'] = new_row['hour'] % minutes_per_day
            new_row['day'] +=1
        new_row['battery_level'] += average_cons
        data_set = data_set.append(new_row, ignore_index=True)
    
    return data_set


def parse_data_set(df_logs, last_row=None):
    '''
    This function creates a dataframe with the format of a dataset.
    The column names are given in the list 'columns'.
    Specifically:
    - milliseconds:     time in millis when the event ocurred.
    - day: day, expressed by a number between 0 and 6, when the event ocurred.
    - hour: hour, expressed by a number between 0 and 23, when the event ocurred.
    - battery_level: level of the battery, the value ranges between 0-100.
    - battery_plugged: whether the mobile is connected to a battery supply or not, 
                    its values can be either 0 or 1.
    - bright_level: level of screen's brightness.
    - screen_on: binary value, indicating whether the screen is on or not.
    - conn: indicates whether the mobile is connected to the network using the 
            mobile connection or not.
    - conn_wifi: indicates if the mobile is conected to some wifi network.
    - temperature: indicates the temperature of the mobile's battery.
    - voltage: indicates the battery's voltage
    - batter_level_grouped: This is the category to which this battery level
                            belongs. There are ten groups:
                        0-15 16-25 26-35 36-45 46-55 56-65 66-75 76-85 86-95 96-100
    At the end of this block, the variable data_set will have all the dataset
    formatted and with all the necessary information.
    '''
    columns = ["milliseconds","day", "hour", "battery_level",\
                "battery_plugged", "bright_level", "screen_on", "conn",\
                "conn_wifi", "temperature", "voltage", "battery_level_grouped"]

    data_set = pd.DataFrame(columns=columns)
    
    if not last_row:
        row = dict()
        for key in columns:
            row[key] = float('nan')
    else:
        row = last_row.copy()
        row["hour"] = to_minutes_wo_day(df_logs[:1].hour.values[0])
        data_set = fill_gaps(data_set, row, last_row)
    
    row["hour"] = to_minutes_wo_day(df_logs[:1].hour.values[0])
    row["day"] = int(df_logs[:1].day.values[0])
    row["milliseconds"] = int(df_logs[:1].milliseconds.values[0])
    last_row = row.copy()
    
    for index, log in df_logs.iterrows():
        different_battery = ((log.feature == "power|battery|level"))
                         
        if (different_battery):
            data_set = data_set.append(row, ignore_index=True)        
            new_row = dict()
            for key in columns:
                new_row[key] = row[key]
            last_row = row
            row = new_row
            row["hour"] = to_minutes_wo_day(log.hour)
            row["day"] = int(log.day)
            
            data_set = fill_gaps(data_set, row, last_row)
            
            row["milliseconds"] = int(log.milliseconds)
            if len(data_set) > 1:
                previous_milli = int(data_set[-1:]["milliseconds"])
                if (previous_milli > row["milliseconds"]):
                    row["milliseconds"] = int(log.milliseconds) + previous_milli
        
        if (log.feature == "screen|brightness|level"):
            row["bright_level"] = int(log.value)
        elif (log.feature == "conn|mobile_supl|detailedstate"):
            if (log.value == "CONNECTED"):
                row["conn"] = 1
            else:
                row["conn"] = 0
        elif (log.feature == "conn|mobile_mms|detailedstate"):
            if (log.value == "CONNECTED"):
                row["conn"] = 1
            else:
                row["conn"] = 0
        elif (log.feature == "conn|mobile|detailedstate"):
            if (log.value == "CONNECTED"):
                row["conn"] = 1
            else:
                row["conn"] = 0
        elif (log.feature == "conn|WIFI|detailedstate"):
            if (log.value == "CONNECTED"):
                row["conn_wifi"] = 1
            else:
                row["conn_wifi"] = 0
        elif (log.feature == "power|battery|level"):
            row["battery_level"] = float(log.value)
            row["battery_level_grouped"] = get_battery_category(int(log.value))
        elif (log.feature == "power|battery|plugged"):
            row["battery_plugged"] = int(log.value)
        elif (log.feature == "screen|power"):
            if ("on" in log.value):
                row["screen_on"] = 1
            else:
                row["screen_on"] = 0    
        elif (log.feature == "power|battery|temperature"):
            row["temperature"] = int(log.value)
        elif (log.feature == "power|battery|voltage"):
            row["voltage"] = int(log.value)
    
    #Drop the nan values        
    cleant = data_set.dropna(axis=0)
    if cleant.shape[0] > 0:
        data_set = cleant
    else:
        data_set = data_set.fillna(-1, axis=0)
    
    return data_set


def get_smoothed_battery_levels(data_set):
    '''
    In this function we smooth the battery_level in order to diminish the 
    noise in the data. We provide two separate models, basic smooth and
    exponential smooth.
    '''
    smoothed = np.array(data_set[:-5]) + \
               np.array(data_set[1:-4]) + \
               np.array(data_set[2:-3]) + \
               np.array(data_set[3:-2]) + \
               np.array(data_set[4:-1]) + \
               np.array(data_set[5:])
    
    #Basic smoothering method
    smoothed = smoothed / 6.
    #Exponential smoothering method
    factor = 2/7.
    exponential = smoothed[:-1] * factor + smoothed[1:] * (1 - factor)
    
    #smoothed = map(lambda x: int(x), smoothed)
    #exponential = map(lambda x: int(x), exponential)
    
    return (smoothed, exponential)    


def plot_smooth_versions(data_set, smoothed, exponential):
    '''
    We plot the consumption along the time, in the three versions: noisy, with 
    basic smoothering and exponential smoothering.
    '''
    fig = plt.figure()
    
    subs = fig.add_subplot(3,1,1)
    subs.plot(data_set['milliseconds'][5:]/60000, \
             data_set['battery_level'][:-5])
    subs.set_title('Not smoothed')
    subs.axes.get_xaxis().set_visible(False)
    
    subs = fig.add_subplot(3,1,2)
    subs.plot(data_set['milliseconds'][5:]/60000, \
             smoothed[:])
    subs.set_title('Simple')
    subs.axes.get_xaxis().set_visible(False)
    
    subs = fig.add_subplot(3,1,3)
    subs.plot(data_set['milliseconds'][6:]/60000, \
             exponential[:])
    subs.set_title('Exponential')
    subs.axes.get_xaxis().set_visible(False)
    
    plt.show()


def add_time_columns(data_set):
    '''
    This function adds the columns relative_min, relative_day and no_day.
    '''
    days = np.array(data_set['day'])
    
    days_range = map(lambda x: x+7 if x<0 else x, np.subtract(days[1:], days[:-1]))
    data_set['no_day'] = np.cumsum(np.append([0], days_range))
    minutes_per_day = 60 * 24
    data_set['relative_day'] = map(lambda day, minu: day + float(minu) / minutes_per_day,\
                                data_set['no_day'], data_set['hour'])
    data_set['relative_min'] = map(lambda day, minu: day * minutes_per_day+ minu,\
                                data_set['no_day'], data_set['hour'])

    return data_set

#%%
def create_dataset():
    
    last_six = []
    last_row = None
    no_logs = 0
    no_rows = 0
    df_logs = pd.read_csv(path, delimiter=';', names=columns, chunksize=CHUNK_SIZE)    
    try:    
        os.remove(path_to_write)
    except:
        print 'File not found'
    
    for chunk in df_logs:
        print 'Configuring date information..'
        data_set = chunk
        data_set = data_set[data_set["timestamp"] != "(invalid date)"]
        data_set = data_set[map(matches, data_set["feature"])]
        data_set["day"] = map(get_day_of_week, data_set["timestamp"])
        data_set["hour"] = map(lambda timestamp: timestamp[11:16], data_set["timestamp"])
        no_logs += data_set.shape[0]
        
        print 'Parsing the logs..'
        data_set = parse_data_set(data_set, last_row)
        last_row = data_set[-1:].to_dict(orient='records')[0]
                
        print 'Smoothing the battery level..'
        if len(last_six) > 0:
            battery_levels = np.append(last_six, data_set['battery_level'])
            smoothed, exponential = get_smoothed_battery_levels(battery_levels)
            last_six = np.array(data_set['battery_level'][-6:])
        else:
            smoothed, exponential = get_smoothed_battery_levels(data_set['battery_level'])
            last_six = np.array(data_set['battery_level'][-6:])
            data_set = data_set[6:]
        #plot_smooth_versions(data_set, smoothed, exponential)
        
        if data_set.shape[0] ==  len(exponential):
            data_set['battery_common_smooth'] = smoothed[:-1]
            data_set['battery_exp_smooth'] = exponential
        else:
            data_set['battery_common_smooth'] = data_set['battery_level']
            data_set['battery_exp_smooth'] = data_set['battery_level']
        
        print 'Adding relative mins and days..'
        data_set = add_time_columns(data_set)
        
        #data_set = data_set.astype(int)
        data_set = data_set.dropna(axis=0)
        no_rows += data_set.shape[0]
        data_set.to_csv(path_to_write, mode='a', index=False, header=False)
        
        print 'Chunk finished!\n\n'
    
    print 'Dataset created!'
    print 'Number of parsed logs', no_logs
    print 'Number of stored rows', no_rows
    
create_dataset()