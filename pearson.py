# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 11:30:47 2018

@author: Aaron
"""
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

def persontest(test1,test2):
    #print("use Pearson,parametric tests test1 and x2")
    r,p=stats.pearsonr(test1,test2)
    print("pearson r**2:",r**2)
    print("pearson r:",abs(r))
    print("pearson p:",p)
    print("----------------------------------")
    return abs(r),p,r**2

selection=['tmax','tmin','rh','nmhc','so2','co','rs','tma','tavg','no','no2','o3','nox','ch4']
x=traindata['maxpm25ma']
relative=[]
for a in selection:
    print("Use Pearson to find relations between 'maxpm25ma' and {}.".format(a))
    y=traindata[a]
    relative.append(persontest(x,y))
    
fig = plt.figure(1,figsize=(15,10))
ax = fig.add_subplot(111)
ax.bar(selection, relative,width=0.5, color="#0066FF")
ax.set_xlabel('Type')
ax.set_ylabel('Correlation Coefficient')
ax.set_title('The correlation coefficient of MaxPM25ma and every type')
fig

pearson=pd.DataFrame(columns=['a','b','r','p','r^2'])
index=0
for a in selection:
    for b in selection:
        print("Use Pearson to find relations between {} and {}.".format(a,b))
        y=traindata[b]
        x=traindata[a]
        r,p,r2=persontest(x,y)
        pearson.loc[index]=[a,b,r,p,r2]
        index=index+1
        
pearson_norepeat=pd.DataFrame(columns=['a','b','r','p','r^2'])
s=pearson.r.duplicated()#find repeat row
index=0
for a in range(0,len(s)):
    if(s[a] and pearson['r'].loc[a]!=1):
        pearson_norepeat.loc[index]=pearson.loc[a]
        index=index+1    
pearson_norepeat=pearson_norepeat.sort_values(by='r',ascending=False) 
name=[]
r=pearson_norepeat['r']
for a in range(0,91):
    name.append(pearson['a'].loc[a]+" vs "+pearson['b'].loc[a])