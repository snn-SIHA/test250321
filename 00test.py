# test

import pandas as pd
import numpy as np
import seaborn as sb
import sklearn as skl
import matplotlib as mpl # 그래프
import matplotlib.pyplot as plt # 그래프 관련
import cv2 as cv # 이미지 처리
import imageio as imio # 영상 처리
from sklearn.preprocessing import MinMaxScaler,StandardScaler # 스케일러
from sklearn.preprocessing import LabelEncoder,OneHotEncoder # 인코더
from sklearn.preprocessing import LabelBinarizer # 멀티클래스 이진 변환
from sklearn.linear_model import LinearRegression,LogisticRegression # 모델 : 선형회귀, 로지스틱
from sklearn.tree import DecisionTreeClassifier # 모델 : 의사결정트리
from sklearn.ensemble import RandomForestClassifier # 모델 : 랜덤포레스트
from sklearn.svm import SVC # 모델 : SVC
from sklearn.datasets import make_classification # 무작위 데이터 생성기
from sklearn.model_selection import train_test_split # 훈련/평가 데이터 분리
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report # 평가 프로세스
from sklearn.metrics import roc_auc_score, roc_curve # ROC,AUC
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
      print('-'*50)
      print(data)
      return
    elif isinstance(data,list):
      data=pd.Series(data)
    print('데이터 정보 출력')
    data.info()
    print(f'행: {data.shape[0]}')
    print('-'*50)
    print(data)
    print(f'{'-'*50}\n{data.value_counts()}')
  except:
    print('>>> 경고! 데이터 형식이 잘못되었습니다!\n>>> checker(data) / repeat= 샘플 출력 횟수')

pathfind = 'C:/Mtest/data/test/nba_source.csv'

#--------------------------------------------------
a = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
print('- '*40)
call(a)
print('='*80)
