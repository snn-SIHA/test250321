# customer

import pandas as pd
import numpy as np
import seaborn as sb
import sklearn as skl
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
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
print(df)

kmeans = KMeans(n_clusters=4,random_state=0)
cluster = kmeans.fit(df)
df['cluster'] = cluster.labels_
df.rename(columns={'mean':'월평균','median':'중앙값','max':'최대값','min':'최소값','membership_period':'회원기간'},inplace=True)

call(df.head())
print('- '*40)

print(df.groupby('cluster').count())
print('- '*40)
'''
          월평균   중앙값   최대값   최소값  회원기간
cluster
0        1334  1334  1334  1334  1334
1         771   771   771   771   771
2        1249  1249  1249  1249  1249
3         838   838   838   838   838
'''
print('TEST')
# 시각화
X = df.copy()
X.drop('cluster',axis=1,inplace=True)
pca = PCA(n_components=2)
pca.fit(X)
pca_X = pca.transform(X)
pca_df = pd.DataFrame(pca_X)
pca_df['cluster'] = df['cluster']
print(pca_df)
print('- '*40)

fig,ax = plt.subplots(1,2,figsize=(12,5))
sb.scatterplot(pca_df,x=0,y=1,hue='cluster',palette='Set1',ax=ax[0])

for i in (pca_df ["cluster"].unique()):
    tmp = pca_df.loc[pca_df["cluster"] == i]
    plt.scatter(tmp[0], tmp[1])
plt.legend(sorted(pca_df['cluster'].unique()))
plt.show()

dfo = pd.read_csv(pathjoin)
df_join = pd.concat([df,dfo],axis=1)
ndf = df_join.groupby(['cluster','is_deleted'],
                      as_index=False).count()[['cluster','is_deleted','customer_id']]

print(ndf)
print('- '*40)
'''
   cluster  is_deleted  customer_id
0        0           0          791
1        0           1          543
2        1           1          771
3        2           0         1231
4        2           1           18
5        3           0          820
6        3           1           18
'''

de0 = ndf.is_deleted==0
de1 = ndf.is_deleted==1

print('탈퇴/지속회원 비율 계산')
for i in range(4):
  temp = ndf.cluster==i
  cal = round((ndf.loc[(temp&de1),'customer_id'].sum() / ndf.loc[(temp),'customer_id'].sum())*100 ,2)
  print(f'\nCluster {i}의 탈퇴회원 비율은 {cal}%')
  print(f'Cluster {i}의 지속회원 비율은 {100-cal}%')

'''
클러스터 1: 탈퇴회원만 존재함
클러스터 2, 3: 지속회원>> 탈퇴회원
클러스터 0: 지속회원과 탈퇴회원 수 비슷

Cluster 0의 탈퇴회원 비율은 40.7%
Cluster 0의 지속회원 비율은 59.3%

Cluster 1의 탈퇴회원 비율은 100.0%
Cluster 1의 지속회원 비율은 0.0%

Cluster 2의 탈퇴회원 비율은 1.44%
Cluster 2의 지속회원 비율은 98.56%

Cluster 3의 탈퇴회원 비율은 2.15%
Cluster 3의 지속회원 비율은 97.85%
'''

print('='*80)
