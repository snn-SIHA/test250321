# test

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

# 사이킷럿(scikit-learn) 라이브러리 임포트 
#--------------------------------------------------------------------------------------------
from datetime import datetime
from dateutil.relativedelta import relativedelta
my_date = datetime(2025,1,1)
new_date1 = my_date + relativedelta(months = 4)
new_date2 = my_date + relativedelta(months = -4)

print(str(new_date1.date())) #2025-05-01
print(str(new_date2.date())) #2024-09-01

print('='*80)
