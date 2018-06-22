# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 16:53:43 2018

@author: Aaron
"""
import matplotlib.pyplot as plt
def mlp_ann(del_features):
    #print(features_name)
    print(del_features)
    #traindata.describe().transpose()
    #vdata2017.describe().transpose()
    y_test=vdata['maxpm25ma'].astype('int')
    X_test=vdata.drop(del_features,axis=1)
    y=traindata['maxpm25ma'].astype('int')
    x=traindata.drop(del_features,axis=1)
    #from sklearn.model_selection import train_test_split
    #X_train, X_test, y_train, y_test = train_test_split(x, y)
    X_train=x
    y_train=y
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(9,9,9),max_iter=500)
    #mlp = MLPClassifier(solver='sgd',activation='logistic',learning_rate_init=0.001,hidden_layer_sizes=(50,50,50,50),max_iter=1500,verbose=True)
    mlp.fit(X_train,y_train.astype('int'))
    predictions = mlp.predict(X_test)
    from sklearn.cross_validation import train_test_split, cross_val_score
    #print('准确率： %s' % cross_val_score(, x, y, cv=5).mean())
    from sklearn.metrics import classification_report,confusion_matrix
    
    
    test_judge=[]
    predict_judge=[]
    x=[]
    ind=0
    
    for a in y_test:
        x.append(ind)
        ind=ind+1
        if a<=35:
            test_judge.append('LOW')
        elif a>35 and a<=53:
            test_judge.append('MID')
        elif a>=54 and a<=70:
            test_judge.append('HIGH')
        else:
            test_judge.append('SUPER')
    for b in predictions:
        if b<=35:
            predict_judge.append('LOW')
        elif b>35 and b<=53:
            predict_judge.append('MID')
        elif b>=54 and b<=70:
            predict_judge.append('HIGH')
        else:
            predict_judge.append('SUPER')
            
    total=0
    match=0
    print(len(predict_judge))
    print(len(test_judge))
    for ind in range(0,len(predict_judge)):
        if predict_judge[ind]==test_judge[ind]:
            match=match+1
            total=total+1
        else:
            total=total+1
    correct=match/total
    print(correct)
    r=y_test
    p=predictions
    fig = plt.figure(1,figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.plot(x,p,color="#0066FF",label='Predict')
    ax.plot(x,r,color="#00DD77",label='Real')
    ax.legend(loc='upper right')
    ax.set_xlabel('Time')
    ax.set_ylabel('PM2.5')
    ax.set_title('Compare real value and predicted value with ANN'+" : "+str(correct*100)+'%')
    fig.show()
    fig.savefig(str(correct*100)+".jpg")
    return correct,predictions

testcase_name=[]
test_result=[]
predict=[]

delfeatures=['maxpm25ma','day']
selection=['tmax','tmin','tma','ch4','rh','no2','nox','rh']
#delfeatures.append(selection[0])
#delfeatures.append(selection[1])
#delfeatures.append(selection[2])
delfeatures.append(selection[3])
#delfeatures.append(selection[4])
#delfeatures.append(selection[5])
#delfeatures.append(selection[6])
delfeatures.append(selection[7])
name=str(vdata.drop(delfeatures,axis=1).columns)
res,predict=mlp_ann(delfeatures)
testcase_name.append(name)
test_result.append(res)

for a in range(0,len(testcase_name)):
    print(testcase_name[a]+":"+str(test_result[a]))
