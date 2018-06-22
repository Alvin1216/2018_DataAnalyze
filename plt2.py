# -*- coding: utf-8 -*-
"""
Created on Tue May  8 10:06:07 2018

@author: user
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from random import randint

#(1)Read the data file: 2001-20170217TWEPA PM10 daily_interp.xlsx
data=pd.read_excel("homework_0508.xlsx")
data.head()
fig=plt.figure(figsize=(30,10))
a1=fig.add_subplot(2,2,1)
a2=fig.add_subplot(2,2,2)
a3=fig.add_subplot(2,2,3)
a1.plot(data["古亭"],color="r")
a2.plot(data["中壢"],color="g")
a3.plot(data["崙背"],color="b")
gutin=data["古亭"]
chunli=data["中壢"]
lunba=data["崙背"]
gutin.describe()
chunli.describe()
lunba.describe()

#plot-001
#The PM10 concentration (μg/m3 ) over time in古亭、中壢、崙背, together in a single figure
df_com=pd.concat([data["古亭"],data["中壢"],data["崙背"]],axis=1)
df_com.columns = ['Guting','Zhongli','Lunbei']
#df_com.columns = ['古亭','中壢','崙背']
df_com.describe()
df_com.plot(figsize=(100,5),ylim=[7,200])
#df_com['Guting'].replace(df_com['Guting'].max(),58,inplace=True)

#plot-002
#Histogram plot for each station
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签  
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
color_set=['#66DD00','#FF8800','#FF0000','#0066FF','#9900FF','#FFFF00']
for a in data:
    plt.clf()
    str_title=a+"站_數據概況(直方圖)"
    file_path="C:/Users/user/Desktop/fig/"+a+".png"
    plt.style.use('ggplot')
    plt.hist(data[a],bins =[0,10,20,30,40,50,60,70,80,90,100],color =color_set[randint(0,5)],edgecolor ='k',label =str_title)
    plt.title(str_title)
    plt.xlabel('濃度分布')
    plt.ylabel('累積數目')
    plt.savefig(file_path, bbox_inches='tight')
    plt.clf()#把圖清空
    print(a)


#plot003
#Scatter plot:中壢VS.崙背
plt.clf()
plt.figure(figsize=(10,10))
plt.scatter(data["中壢"],data["崙背"],s=30,c='steelblue',marker='.',alpha=0.5)
#前面兩個分別為xy座標,s是點的大小,c是顏色,marker是點的樣子,alpha是亮度
plt.title('中壢 vs 崙背')
plt.xlim(0,250)#設定顯示範圍
plt.ylim(0,250)
plt.xlabel('中壢')
plt.ylabel('崙背')
plt.show()
plt.savefig("中壢崙背003.png", bbox_inches='tight')

#spx = df_com['Guting']
#spx.plot(ax=ax,figsize=(30,5),style='k-')
#spx.plot(figsize=(50,5),xlim=['1/1/2001','12/31/2001'],ylim=[7,200],kind="hist")
#ax.set_xlim(['1/1/2007', '12/31/2007'])
#ax.set_ylim([0, 90])
#ax.set_title('dates in the 2007')

#Descriptive statistics001
#Calculate mean and standard deviation value of 古亭、中壢、崙背, and explain the phenomenon
df_com=pd.concat([data["古亭"],data["中壢"],data["崙背"]],axis=1)
df_com.describe()

#Descriptive statistics001
#Calculate Correlation coefficient between中壢VS.崙背
df2=pd.concat([data["中壢"],data["崙背"]],axis=1)
df2.corr() 