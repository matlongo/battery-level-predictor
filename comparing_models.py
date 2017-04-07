# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:17:32 2016

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
import seaborn

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
'''
Functions to be used..
'''
def load_dataset(path_dataset):
    data_set = pd.read_csv(path_dataset, names=columns)#, dtype=types)
    
    return data_set


def find_pos_by_day(data_set, day, hour):
    days = np.where((data_set['day']==day) & (data_set['hour'] == hour))[0]
                                           
    length = min([len(days), 20])
    return days[:length]
    

def mean_square_error(y_true, y_hat):
    error = np.subtract(y_true, y_hat)
    square_error = np.multiply(error, error)
    
    return round(np.sqrt(sum(np.divide(square_error, len(error)))), 2)
    
def generating_stats(data_set, models, day, hour, user, initial_pos, steps=100):
    fin = initial_pos + steps
    mins = np.array(data_set['relative_min'][initial_pos:fin])
    y_true = np.array(data_set['battery_exp_smooth'][initial_pos:fin])
    
    paper_pred = models[0].predict_level(mins, initial_pos=initial_pos)
    
    pro_pred = models[1].predict_level(mins, initial_pos=initial_pos)
    
    #regr_pred_simple, _ = battery_regression.predict_level(initial_pos=initial_pos,\
    #                                           steps=steps, other_models=False)
    
    regr_pred, _ = models[2].predict_level(initial_pos=initial_pos, steps=steps,
                                                    other_models=True)
    
    row = dict()
    row['user'] = user
    row['day'] = day
    row['hour'] = hour
    row['paper'] = mean_square_error(y_true, paper_pred)
    row['pro'] = mean_square_error(y_true, pro_pred)
    #row['r_simple'] = mean_square_error(y_true, regr_pred_simple)
    row['regr'] = mean_square_error(y_true, regr_pred)
    
    #row[('R2', 'paper')] = r2_score(y_true, paper_pred)
    #row[('R2', 'r_other')] = r2_score(y_true, regr_pred)
    row['pos'] = initial_pos
    
    #name = 'user:{0}_day:{1}_hour:{2}_steps:{3}.png'.format(user, day, hour, steps)
    #plot_curves(paper_pred, pro_pred, regr_pred_simple, regr_pred, y_true,
    #            mins, name)
    return row

def plot_stats(stats, users, days, pdf):
    means = stats.groupby(['user', 'day']).mean()[['paper', 'pro', 'r_simple', 'r_other']]
    pos = range(4)
    labels = ['paper', 'pro', 'r_simple', 'r_other']
    rows = len(users)
    cols = len(days)
    
    f, axarr = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8), sharey='row')
    plt.setp(axarr, xticks=np.add(pos, 0.4), xticklabels=labels)
    
    cols_headers = ['Day {}'.format(col) for col in days]
    rows_headers = ['{}'.format(header) for header in users]
    
    for ax, col in zip(axarr[0], cols_headers):
        ax.set_title(col)
    
    for ax, row in zip(axarr[:,0], rows_headers):
        ax.set_ylabel(row, rotation=90, size='large')
        
    for row in range(rows):
        for col in range(cols):
            axarr[row, col].bar(pos, means.loc[(users[row], days[col])])
    
    f.tight_layout()
    plt.savefig(pdf, format='pdf')


def plot_curves(paper_pred, pro_pred, regr_pred_simple, regr_pred, 
                y_true, mins, name):
    methods = ['paper', 'pro', 'r_simple', 'r_other']
        
    f, axarr = plt.subplots(nrows=4, ncols=1, figsize=(12, 8))
    
    rows_headers = ['{}'.format(header) for header in methods]
        
    for ax, row in zip(axarr, rows_headers):
        ax.set_ylabel(row, rotation=90, size='large')
    
    axarr[0].plot(mins, paper_pred, 'r',
                     mins, y_true, 'b')
    axarr[1].plot(mins, pro_pred, 'r',
                     mins, y_true, 'b')
    axarr[2].plot(mins, regr_pred_simple, 'r',
                     mins, y_true, 'b')
    axarr[3].plot(mins, regr_pred, 'r',
                     mins, y_true, 'b')
    
    f.tight_layout()
    f.savefig(name)
    

#%%
users = ['user1', 'user2', 'user3', 'user4', 'user5', 'user6']
days = [2, 4, 6]
hours = [x for x in range(0,1439,60)]
l_steps = [240]
#pp = PdfPages('stats.pdf')

for steps in l_steps:
    stats = pd.DataFrame(columns=['user', 'day', 'paper', 'pro',
                                  'regr', 'pos'])
    for user in users:
        print('Processing {0}'.format(user))
        path_dataset = "../preprocessed/preprocessed_{0}.csv".format(user)#'../preprocessed/preprocessed_{0}.csv'.format(user)
        data_set = load_dataset(path_dataset)
        
        diff = np.subtract(data_set['battery_exp_smooth'][1:], data_set['battery_exp_smooth'][:-1])
        diff = np.array(np.append(diff, [0]))
        plugged = [1 if x > 0 or y == 100 else 0 for (x, y) in zip(diff, data_set['battery_exp_smooth'])]
        data_set['battery_plugged'] = plugged
        
        data_set = data_set.drop_duplicates([u'day', u'hour', u'battery_level', u'battery_exp_smooth'])
        
        # We train the models
        # Paper model
        battery_paper = BatteryPaper()
        battery_paper.fit(data_set)
        # Battery-Pro model
        battery_pro = BatteryPro()
        battery_pro.fit_by_time(data_set, mins=10)
        # Battery regression model
        battery_regression = BatteryRegression()
        battery_regression.fit(data_set)
        
        models = [battery_paper, battery_pro, battery_regression]
        
        for day in days:
            print('--Day {0}'.format(day))
            for hour in hours:
                initial_pos_array = find_pos_by_day(data_set, day, hour)
                for initial_pos in initial_pos_array:
                    print('----Initial pos {0}'.format(initial_pos))
                    # We generate the stats and cumulate them in a DF
                    if initial_pos + steps < data_set.shape[0]:
                        row = generating_stats(data_set, models, day, hour, user, initial_pos, steps)
                        stats = stats.append(row, ignore_index=True)
                
    #plot_stats(stats, users, days, pp)
    
#pp.close()
stats.to_csv('stats.csv')
grouped = stats.groupby(['user', 'day', 'hour'])
print(stats)

'''
data_set = load_dataset(path_dataset)
diff = np.subtract(data_set['battery_exp_smooth'][1:], data_set['battery_exp_smooth'][:-1])
diff = np.array(np.append(diff, [0]))
plugged = [1 if x > 0 or y == 100 else 0 for (x, y) in zip(diff, data_set['battery_exp_smooth'])]
data_set['battery_plugged'] = plugged
battery_regression = BatteryRegression()
battery_regression.fit(data_set[:50000])
pd.DataFrame(np.array([diff, plugged, data_set['battery_exp_smooth']]).T)

'''













