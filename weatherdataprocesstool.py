
import numpy as np
import pandas as pd 
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns

#天氣資料欄位值處理
def dataMean(s1,colName,groupbyIndex=list):
    if s1[colName].max()>1:
        dataMean=s1[s1[colName]<s1[colName].max()]#拿掉沒資料的
    else:
        dataMean=s1
    dataMean=dataMean.groupby(groupbyIndex)[colName].mean()
    return dataMean

def dataSum(s1,colName,groupbyIndex):
    if s1[colName].max()>1:
        dataSum=s1[s1[colName]<s1[colName].max()]#拿掉沒資料的
    else:
        dataSum=s1
    dataSum=dataSum.groupby(groupbyIndex)[colName].sum()
    return dataSum

def dataMax(s1,colName,groupbyIndex):
    if s1[colName].max()>1:
        dataMax=s1[s1[colName]<s1[colName].max()]#拿掉沒資料的
    else:
        dataMax=s1
    dataMax=dataMax.groupby(groupbyIndex)[colName].max()
    return dataMax

def dataMin(s1,colName,groupbyIndex):
    if s1[colName].max()>1:
        dataMin=s1[s1[colName]<s1[colName].max()]#拿掉沒資料的
    else:
        dataMax=s1
    dataMin=dataMin.groupby(groupbyIndex)[colName].min()
    return dataMin

#價格報酬率label建立
def priceReturnFeature(stock_id,route):
    price=pd.read_pickle('/Users/benbilly3/Desktop/資策會專題/rawMaterialPricePrediction/RM_Price/rawMaterialPrice.pickle')
    price=price.loc[stock_id]
    price['next']=price['Close'].shift(-route)
    price['return']=price['next']/price['Close']
    price=price.dropna()
    return price


#機器學習視覺化

def plot_decision_regions(X, y, classifier, resolution=0.02):# classifier為選取器選擇

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('palegreen','pink', 'palegreen', 'lightskyblue', 'snow', 'lemonchiffon')
    colors2 = ('forestgreen','indianred', 'royalblue', 'gray', 'darkgoldenrod')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    cmap2 = ListedColormap(colors2[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
 

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    color=cmap2(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)
        
#ML stock_Prediction confusionMatrixSheet
def confusionMatrixSheet(y_test,y_pred,classifier):

    TP=sum([1 if (a==1)&(b==1) else 0 for a,b in zip(y_test,y_pred)])/(list(y_test).count(1))
    FP=sum([1 if (a==1)&(b==0) else 0 for a,b in zip(y_test,y_pred)])/(list(y_test).count(1))
    TN=sum([1 if (a==0)&(b==0) else 0 for a,b in zip(y_test,y_pred)])/(list(y_test).count(0))
    FN=sum([1 if (a==0)&(b==1) else 0 for a,b in zip(y_test,y_pred)])/(list(y_test).count(0))
    data={
    '跟上漲月率':round(TP*100,2),
    '錯過漲月率':round(FP*100,2),
    '跟上跌月率':round(FN*100,2),
    '躲過跌月率':round(TN*100,2),
    '精準度':round((TP+TN)/(TP+FP+TN+FN)*100,2)
    }
    
    # confusionMatrixPlot
    plt.rcParams['font.family']=['Arial Unicode MS']
    plt.rcParams['figure.figsize'] = (12, 12)
    plt.rcParams['font.size']=18
    s1=pd.Series([TP,FN],index=['Up','Down'])
    s2=pd.Series([FP,TN],index=['Up','Down'])
    data2={
    'Up':s1,
    'Down':s2
    }
    df=pd.DataFrame(data2)
    
    ax = plt.axes()
    sns.heatmap(df, square=True,vmax=1.0, linecolor='white', annot=True, cmap="YlGnBu")
    plt.title(classifier+'_ConfusionMatrix',fontsize='x-large')
    plt.xlabel('Predict_label')
    plt.ylabel('True_label')
    
    return pd.DataFrame(data,index=[classifier])