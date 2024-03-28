import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv('./default of credit card clients.csv')

df.head()

# df.info()

df.columns

indexes=df.iloc[0,:].values
indexes

df_copy=pd.DataFrame(df.values[1:],columns=indexes)
df_copy.head()

df_copy.shape

# df_copy.info()

df_copy.iloc[0,:].values

def cat_cols(df):
    arr=[]
    for i in df.columns:
        if df[i].dtype=='O':
            arr.append(i)
    return arr

categorical_columns=cat_cols(df_copy)
# print(categorical_columns)

def cat_to_num(df,cat_cols):
    for i in cat_cols:
        df[i]=df[i].astype(float)

cat_to_num(df_copy,categorical_columns)
# df_copy.info()

df_copy.drop(labels=['ID'],axis=1,inplace=True)
df_copy.head()

df_copy.describe()

columns=df_copy.columns
columns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

df_copy['default payment next month'].value_counts()

imb_df=df_copy.copy()

from imblearn.over_sampling import SMOTE

smote=SMOTE()

resample_X,resample_Y=smote.fit_resample(imb_df.iloc[:,:-1],imb_df.iloc[:,-1])

resample_X.shape

resample_Y.shape

balanced_df=pd.concat([resample_X,resample_Y],axis=1)

balanced_df

balanced_df['default payment next month'].value_counts()

X=balanced_df.iloc[:,:-1]
y=balanced_df['default payment next month']

xtrain,xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25,random_state=45)

xtrain.shape,xtest.shape,ytrain.shape,ytest.shape

ytrain.value_counts()

ytest.value_counts()

scaler=StandardScaler()
trans_xtrain=scaler.fit_transform(xtrain)
trans_xtrain

trans_xtest=scaler.transform(xtest)
trans_xtest

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

raw_models={
    'RandomForestClassifier':RandomForestClassifier(),
#     'SVC':SVC(),
#     'KNeighborsClassifier':KNeighborsClassifier()
}

from sklearn.metrics import accuracy_score

def best_model(ml_models,xtrain,xtest,ytrain,ytest):
    performance=dict()
    models=[]
    idx=0
    for key in ml_models.keys():
        arr=[]
        arr.append(idx)
        idx+=1
        model=ml_models[key]
        model.fit(xtrain,ytrain)
        y_pred=model.predict(xtest)
        score=accuracy_score(ytest,y_pred)
        arr.append(score)
        performance[key]=arr
        models.append(model)
    return performance,models

performance,trained_models=best_model(raw_models,trans_xtrain,trans_xtest,ytrain,ytest)

performance

trained_models

def find_best_model(performance,Models):
    arr=sorted(performance,key=lambda x:performance[x][1],reverse=True)
    key=arr[0]
    idx=performance[key][0]
    return Models[idx]

clf_model=find_best_model(performance,trained_models)
clf_model

print(clf_model.score(trans_xtest,ytest))

import pickle

pickle.dump(scaler,open('scaler.pkl','wb'))

pickle.dump(clf_model,open('model.pkl','wb'))