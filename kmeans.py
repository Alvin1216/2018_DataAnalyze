# -*- coding: utf-8 -*-
"""
Created on Sun May 27 17:52:42 2018

@author: Aaron
"""

import numpy as np
import pandas as pd
import random,math

def kmeans(data,k=3):
    #input datafram-->datafram  k-->int 沒有輸入 預設為三
    p=data
    colorlist=['#FF0000','#0066FF','#00FF00','#FFFF00','#FF8800']
    pointlist=random.sample(range(0,len(p)), k)
    #先從原本的資料集選k個行星出來
    for s in range(0,k):
        p['group'][pointlist[s]]=str(s)
        for a in range(0,len(p)):
            distancelist=[]
            for s in range(0,k):
        #把這個點到各行星的最小距離算出來
                distance=(p['x'][a]-p['x'][pointlist[s]])**2+(p['y'][a]-p['y'][pointlist[s]])**2
                distance=math.sqrt(distance) 
                distancelist.append(distance)
            #選那個最小距離
            group=distancelist.index(min(distancelist))
            p['group'][a]=p['group'][pointlist[group]]
    
    newpoint=[]
    for f in range(0,k):
        #找出新的點
         x=p[p['group']==str(f)].sum()['x']
         y=p[p['group']==str(f)].sum()['y']
         d={'x':int(x/len(p[p['group']==str(f)])),'y':int(y/len(p[p['group']==str(f)]))}
         newpoint.append(d)
         
     
    #找出新的點之後 後面就作重複的事情
    print("Selected automatically Centroid:")
    formerpoint=[]
    while(newpoint!=formerpoint):
        formerpoint=newpoint
        for a in range(0,len(p)):
            distancelist=[]
            for s in range(0,k):
                #把這個點到各行星的最小距離算出來
                distance=(p['x'][a]-formerpoint[s]['x'])**2+(p['y'][a]-formerpoint[s]['y'])**2
                distance=math.sqrt(distance)
                distancelist.append(distance)
                #選那個最小距離
            group=distancelist.index(min(distancelist))
            p['group'][a]=str(group)
        
        newpoint=[]
        for f in range(0,k):
            #找出新的點
            x=p[p['group']==str(f)].sum()['x']
            y=p[p['group']==str(f)].sum()['y']
            d={'x':int(x/len(p[p['group']==str(f)])),'y':int(y/len(p[p['group']==str(f)]))}
            newpoint.append(d)
        print(newpoint)
    print("finished!")
    for r in range(0,len(p)):
        p['color'][r]=colorlist[int(p['group'][r])]
    
    return p

#隨機作出兩組數據 組成xy坐標
datax=np.random.randint(10,size=80)
datay=np.random.randint(10,size=80)
datay=list(datay)
datax=list(datax)
data={'x':datax,'y':datay};
p=pd.DataFrame(data);
p['color']='#00BBFF'
p['group']='-1'
p.plot.scatter(x='x',y='y',c=p['color'])
p=kmeans(p,3)
p.plot.scatter(x='x',y='y',c=p['color'])
    