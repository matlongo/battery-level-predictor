# _*_ coding: utf_8 _*_
"""
Created on Wed Feb 15 23:25:19 2017

@author: matlongo
"""

#%%
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn

#%%
columns = [u'milliseconds', u'day', u'hour', u'battery_level', u'battery_plugged',
       u'bright_level', u'screen_on', u'conn', u'conn_wifi', u'temperature',
       u'voltage', u'battery_level_grouped', u'battery_common_smooth',
       u'battery_exp_smooth', u'no_day', u'relative_day', u'relative_min']

df = pd.read_csv(, names=columns)

#%%
plt.figure(figsize=(8, 3))
plt.plot(df['battery_level'][10000: 13000])
plt.xticks([])


#%%
corr = np.corrcoef(df, rowvar=0)
corr_df = pd.DataFrame(corr, columns=columns, index=columns)
corr_df.to_csv('correlation.csv')

#%%
df['hour2'] = map(int, df['hour'] / 60)
means = df[['hour2', 'battery_level']].groupby('hour2').mean()
plt.figure(figsize=(8, 3))
plt.bar(range(24), means.values[:-1])
plt.xticks(range(25))

#%%
f, ax = plt.subplots(3, 1)
ax[0].plot(df['voltage'][:1000])
ax[1].plot(df['battery_level'][:1000])
ax[2].plot(df['battery_plugged'][:1000])

#%%
aux = df[['day', 'hour2', 'battery_level']].groupby(['day', 'hour2']).mean().reset_index()
f, ax = plt.subplots(7, 1, sharey=True, sharex=True, figsize=(8, 8))

for d in range(7):
    ax[d].plot(range(24), aux[aux['day'] == d].sort_values('hour2')['battery_level'][:-1])

#%%
aux2 = df[['screen_on', 'battery_plugged', 'conn']].groupby(['screen_on', 'battery_plugged']).sum().reset_index()
pivot = aux2.pivot_table(index='battery_plugged', columns='screen_on', values='conn').iloc[:2]
pivot = np.array(pivot)

# Get some pastel shades for the colors
colors = plt.cm.BuPu(np.linspace(0, 0.5, len(pivot)))
colors = colors[::-1]
n_rows = len(pivot)
columns = ['Screen off', 'Screen on']

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.array([0.0] * len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, pivot[row], bar_width, color=colors[row])
    print row, colors[row]

plt.table(cellText=np.array(pivot),rowLabels=['battery unplugged', 'battery plugged'],
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom')


plt.subplots_adjust(left=0.2, bottom=0.2)

#plt.ylabel("Loss in ${0}'s".format(value_increment))
#plt.yticks(values * value_increment, ['%d' % val for val in values])
plt.xticks([])
#plt.title('Loss by Disaster')










