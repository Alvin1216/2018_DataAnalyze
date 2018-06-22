# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 17:52:55 2018

@author: Aaron
"""
#col 是直的
#row 是橫的
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
def linearRegression_sklearn(x,y):
    reg=linear_model.LinearRegression()
    reg.fit(x,y)
    print(reg.coef_)
    print(reg.intercept_)
def linearRegression_statsmodels(x,y):
    #x = sm.add_constant(x)
    model=sm.OLS(y.astype(float),x.astype(float)).fit()
    print(model.summary())
def persontest(test1,test2):
    print("use Pearson,parametric tests test1 and x2")
    r,p=stats.pearsonr(test1,test2)
    print("pearson r**2:",r**2)
    print("pearson p:",p)
def judgestamp(df,startindex=0):
    sdata_marker=df.copy()
    sdata_marker['judge']=0
    for ind in range(startindex,len(sdata_marker['maxpm25ma'])+startindex):
        print(str(ind-1)+" is finished!")
        if(sdata_marker['maxpm25ma'].loc[ind]<=35):
            sdata_marker['judge'].loc[ind]='LOW'
            continue
        elif(sdata_marker['maxpm25ma'].loc[ind]>35 and sdata_marker['maxpm25ma'].loc[ind]<=53):
            sdata_marker['judge'].loc[ind]='MID'
            continue
        elif(sdata_marker['maxpm25ma'].loc[ind]>=54 and sdata_marker['maxpm25ma'].loc[ind]<=70):
            sdata_marker['judge'].loc[ind]='HIGH'
            continue
        else:
            sdata_marker['judge'].loc[ind]='SUPER'
            continue
    return sdata_marker
def GroupbyAndCreatNewDataframe(original):
    #把資料照日期分組 拉出需要的東西
    sdata=pd.DataFrame(columns=['day','maxpm25ma','tavg','tma','tmax','tmin','rh','rs','so2','co','nmhc','no','no2','o3','nox','ch4'])
    sdataRowIndex=0
    for dataindex in range(0,len(original)):
        group = original[dataindex].groupby('time')
        for x in group:
            #turple不能修改內容
            #先轉list再轉回來
            try:
                t=list(x)
                a=t[1]
                col=list(a['testitems'])#行名(columns)是檢測種類的名字 先存
                a.pop('time')#刪掉不用的東西
                a.pop('station')#刪掉不用的東西
                a.pop('testitems')#刪掉不用的東西
                a=a.transpose()#轉置 之後一個colums算比較好算
                a.columns=col#把colums名改掉
                t[1]=a
                tmax=a['AMB_TEMP'].max()
                tmin=a['AMB_TEMP'].min()
                tma=moving_average(list(a['AMB_TEMP'])).max()
                tavg=a['AMB_TEMP'].mean()
                moveavgpm25=moving_average(list(a['PM2.5'])).max()
                rh=moving_average(list(a['RH'])).max()
                rs=moving_average(list(a['WIND_SPEED'])).max()
                co=moving_average(list(a['CO'])).max()
                so2=moving_average(list(a['SO2'])).max()
                nmhc=moving_average(list(a['NMHC'])).max()
                no=moving_average(list(a['NO'])).max()
                no2=moving_average(list(a['NO2'])).max()
                o3=moving_average(list(a['O3'])).max()
                nox=moving_average(list(a['NOx'])).max()
                ch4=moving_average(list(a['CH4'])).max()
                #rh=a['RH'].mean()
                #rs=a['WIND_SPEED'].mean()
                #co=a['CO'].mean()
                #so2=a['SO2'].mean()
                #nmhc=a['NMHC'].mean()
                #算好的東西存進去新的datafram loc拿來定位第幾列
                sdata.loc[sdataRowIndex]=[x[0],moveavgpm25,tavg,tma,tmax,tmin,rh,rs,so2,co,nmhc,no,no2,o3,nox,ch4]
                sdataRowIndex=sdataRowIndex+1
                x=tuple(t)
                print(x[0]+" finished!")
            except:
                #有缺值的再前面dropna整row都掰掰了
                #所以到這邊如果資料不完整就會直接說再見
                print(x[0]+" something wrong!")
    return sdata

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
        

sdata=GroupbyAndCreatNewDataframe(data)


testcase_name=[]
test_result=[]

vdata=sdata.loc[2135:]
traindata=sdata.loc[0:2134]
vdata=judgestamp(vdata,startindex=2135)

##Round start here!
selection_todel=['maxpm25ma','day','tmin','tmax','tma','ch4','rh','no','no2']
selection_pdata_todel=['maxpm25ma','day','tmin','tmax','tma','judge','ch4','rh','no','no2']

traindata_copy=traindata
x=traindata_copy.drop(selection_todel,axis=1)
y=pd.DataFrame(traindata['maxpm25ma'])
reg=linear_model.LinearRegression()
reg.fit(x,y)

#linearRegression_statsmodels(x,y)
#linearRegression_sklearn(x,y)

pdata=vdata.drop(selection_pdata_todel,axis=1)
predictValue=[]
for k in range(2135,2135+len(pdata['tavg'])):
    predictarray=np.array([(list(pdata.loc[k]))])
    predicted_sales = reg.predict(predictarray)
    predictValue.append(predicted_sales)
    print(predicted_sales)

pdata['predict']=0
pdata['predictJudge']=0
index=0
for k in range(2135,2135+len(pdata['tavg'])):
    print(index)
    putin=float(predictValue[index][0])
    if(putin<=35):
        pdata['predictJudge'].loc[k]='LOW'
        pdata['predict'].loc[k]=putin
        index=index+1
        #continue
    elif(putin>35 and putin<=53):
        pdata['predictJudge'].loc[k]='MID'
        pdata['predict'].loc[k]=putin
        index=index+1
        #continue
    elif(putin>=54 and putin<=70):
        pdata['predictJudge'].loc[k]='HIGH'
        pdata['predict'].loc[k]=putin
        index=index+1
        #continue
    else:
        pdata['predictJudge'].loc[k]='SUPER'
        pdata['predict'].loc[k]=putin
        index=index+1
        #continue        
match=0
total=0;
for a in range(0,len(pdata['predict'])):
    if(vdata['judge'].loc[a+2135]==pdata['predictJudge'].loc[a+2135]):
        match=match+1
        total=total+1
    else:
        total=total+1
correct=match/total
print(correct)

fig = plt.figure(1,figsize=(30,10))
ax = fig.add_subplot(111)
ax.plot(x1,bp,color="#0066FF",label='Predict')
ax.plot(x1,real,color="#FF8800",label='Real')
ax.legend(loc='upper right')
ax.set_xlabel('Time')
ax.set_ylabel('PM2.5')
ax.set_title('Compare real value and predicted value')
fig
#testcase_name.append(str(pdata.columns[:-2]))
#test_result.append(correct)
#delete_selection.append(str(selection_todel))
#vdata=judgestamp(vdata)

#print("--------------------All data--------------good---")
#x=traindata_copy.drop(['maxpm25ma','day','tmax','tmin'],axis=1)
#y=pd.DataFrame(traindata['maxpm25ma'])
#linearRegression_statsmodels(x,y)
#linearRegression_sklearn(x,y)
