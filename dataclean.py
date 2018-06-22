# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 11:58:18 2018

@author: Aaron
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import scipy.stats as stats
import statsmodels.api as sm

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False
def isint(value):
  try:
    int(value)
    return True
  except ValueError:
    return False
def isnan(value):
    if value=='nan':
        return True
    else:
        return False
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


dlist=['2010.csv','2011.csv','2012.csv','2013.csv','2014.csv','2015.csv','2016.csv','2017.csv']
data=[]
for i in range(0,len(dlist)):
    data.append(pd.read_csv("./airdata/"+dlist[i]))

for i in range(1,10):
    f=str(i)
    change=str('0'+str(i))
    a={f:change}
    data[0].rename(columns=a,inplace=True)
    #Whether to return a new %(klass)s. If True then value of copy is ignored.
    #inplace是三小我也不知道 不過不加上去的話改不了
    #inplace是直接改掉data

#把非數字都挖出來
#全部清0
#nan好像無法清空--->解決 用datafram_name.fillna(0)
nanCount=0
for i in range(0,len(data)):
    #橫的有幾行 row行
    for row in range(0,len(data[i])):
        #直的從第四行開始是數字 一直到最後
        for colums in range(3,27):
            if(isint(data[i].iat[row,colums])):
                data[i].iat[row,colums]=int(data[i].iat[row,colums])
            elif(isfloat(data[i].iat[row,colums])):
                data[i].iat[row,colums]=float(data[i].iat[row,colums])
            else:
                data[i].iat[row,colums]=int(0)
    print("Finish one csv~")
    #data[i].fillna(0, inplace=True)
    #fillna好像沒用 解法2 isnull
    data[i].dropna(how="any", inplace=True)
    #blank=np.where(pd.isnull(data[2]))
    #for blanker in range(0,len(blank[0])):
        #data[i].iat[blank[0][blanker],blank[1][blanker]]=int(0)