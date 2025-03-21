# customer

import pandas as pd
import numpy as np
import seaborn as sb
import sklearn as skl
import matplotlib as mpl # 그래프
import matplotlib.pyplot as plt # 그래프 관련
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from matplotlib import font_manager
from matplotlib import rc
from tabulate import tabulate

mpl.rc('axes',unicode_minus=False)
font = font_manager.FontProperties(fname = "c:/Windows/Fonts/malgun.ttf").get_name()
rc('font',family=font)

def call(data,style=None):
  if style == None:
    print(tabulate(data,headers='keys',tablefmt='github'))
  else:
    print(tabulate(data,headers='keys',tablefmt=style))

def checker(data):
  try:
    if isinstance(data,pd.DataFrame):
      print('데이터 정보 출력')
      data.info()   
      print(f'행: {data.shape[0]}, 열: {data.shape[1]}')
      print('-'*80)
      print(data)
      return
    elif isinstance(data,list):
      data=pd.Series(data)
    print('데이터 정보 출력')
    data.info()
    print(f'행: {data.shape[0]}')
    print('-'*80)
    print(data)
    print(f'{'-'*80}\n{data.value_counts()}')
  except:
    print('>>> 경고! 데이터 형식이 잘못되었습니다!\n>>> checker(data) / repeat= 샘플 출력 횟수')

pathjoin = 'C:/Mtest/data/bike/customer_join.csv'

#--------------------------------------------------
df = pd.read_csv(pathjoin)
print('- '*40)

checker(df)
print('- '*40)

# 데이터 분리
df = df[['mean','median','max','min','membership_period']]
checker(df)
print('- '*40)

# 문제 1
# 스케일링
STS = StandardScaler()
df = pd.DataFrame(STS.fit_transform(df),columns=df.columns)
print('스케일링')
print(df)

kmeans = KMeans(n_clusters=4,random_state=0)
cluster = kmeans.fit(df)

print('='*80)
