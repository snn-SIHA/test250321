import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import rc

# 음수표기 관리
import matplotlib as mpl
mpl.rc('axes', unicode_minus=False)
mpl.rcParams['axes.unicode_minus']=False

font_name = font_manager.FontProperties(fname='c:/windows/Fonts/malgun.ttf').get_name()
rc('font', family=font_name)

import pandas as pd
import numpy as np
import seaborn as sns 
import time

#--------------------------------------------------------------------------------------------
train = pd.read_csv('./bike/train.csv')
test = pd.read_csv('./bike/test.csv')
print(train.info())

train['datetime'] =train.datetime.apply(pd.to_datetime)
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day

print('3-18-화요일 람다식 적용한 경우')
train['date'] = train.datetime.apply(lambda x: x.date) #람다식
print(train['date'])
# train['date'] = train['datetime'].dt.date
print()

# train['dayofweek'] = train['datetime'].dt.day_name() 
train['dayofweek'] = train['datetime'].apply(lambda x: x.day_name())
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['quarter'] = train['datetime'].dt.quarter
print( train )  
print()  #ok 

print('3-18-화요일 람다식 적용한 경우 sns.pointplot그래프')
plt.figure(figsize=(16,6))
sns.pointplot(data=train, x='hour', y='count', hue='dayofweek',  palette='Set1')  #sns.lineplot(data=train, x='date', y='count')
plt.show()


print()
print()