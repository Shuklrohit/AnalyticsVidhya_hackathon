#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as nm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import datetime as dt
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.XGboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections
le = preprocessing.LabelEncoder()
label_encoder = preprocessing.LabelEncoder() 
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


# In[122]:


train = pd.read_csv(r'C:\Hackothon\Analytics Vidhya Hackothon/train.csv')
test = pd.read_csv(r'C:\Hackothon\Analytics Vidhya Hackothon/test.csv')


# In[84]:


train.groupby(['m13']).size()


# In[123]:


train['source']= label_encoder.fit_transform(train['source']) 
train['financial_institution']= label_encoder.fit_transform(train['financial_institution']) 
train['loan_purpose']= label_encoder.fit_transform(train['loan_purpose']) 
test['source']= label_encoder.fit_transform(test['source'].astype(str)) 
test['financial_institution']= label_encoder.fit_transform(test['financial_institution'].astype(str)) 
test['loan_purpose']= label_encoder.fit_transform(test['loan_purpose'].astype(str)) 


# In[124]:


train['origination_date'] =train['origination_date'].str.replace('-', '')
train['first_payment_date'] = train['first_payment_date'].str.replace('/', '')
test['origination_date'] =test['origination_date'].str.replace('/', '')
test['first_payment_date'] = test['first_payment_date'].str.replace('/', '')


# In[87]:


from sklearn.utils import resample


# In[88]:


df_majority = train[train.m13==0]
df_minority = train[train.m13==1]


# In[89]:


df_minority.head(4)


# In[90]:


df_majority.head(4)


# In[126]:


y = train['m13']
X = train.drop(['m13'], axis = 1)


# In[127]:


from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()


# In[128]:


Scaler.fit(X)
Scaler.fit(test)


# In[129]:


scaled_features = Scaler.transform(X)
scaled_features_test = Scaler.transform(test)


# In[96]:


# scaled_features


# In[ ]:





# In[ ]:





# In[97]:


# Upsample minority class
# df_minority_upsampled = resample(df_minority, 
#                                  replace=True,     # sample with replacement
#                                  n_samples=636,    # to match majority class
#                                  random_state=123) # reproducible results

# # Combine majority class with upsampled minority class
# df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
# df_upsampled.balance.value_counts()


# In[92]:


df_upsampled.m13.value_counts()


# In[43]:


# y_test


# In[130]:


# Separate input features (X) and target variable (y)
# y = train.m13
# X = train.drop('m13', axis=1)


# In[66]:


print( np.unique( y_test ) )


# In[139]:


from sklearn.model_selection import train_test_split

# This is explicitly used for undersampling.
X_train, X_test, y_train, y_test = train_test_split(scaled_features, y,  random_state = 1, stratify=y)


# In[140]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier( n_estimators = 10,min_samples_split=10,min_samples_leaf = 1, max_features = 'sqrt',
                             max_depth=None, bootstrap =True)
rfc.fit(X_train, y_train)

 
# Predict on training set
pred_y_1 = rfc.predict(X_test)
 
# Is our model still predicting just one class?
print( np.unique( pred_y_1 ) )
# [0 1]
 
# How's our accuracy?
print( accuracy_score(y_test, pred_y_1) )


# In[141]:


print(recall_score(y_test, pred_y_1))
confusion_matrix(y_test, pred_y_1)


# In[102]:





# # #SMOTE-------

# In[134]:


from imblearn.over_sampling import SMOTE


# In[142]:


X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, random_state = 1, stratify=y)

smt = SMOTE()
X_train, y_train = smt.fit_sample(X_train, y_train)


# In[143]:


np.bincount(y_train)


# In[176]:


from sklearn.ensemble import RandomForestClassifier
rfc1 = RandomForestClassifier( n_estimators = 173,min_samples_split=15,min_samples_leaf = 29, max_features = 'sqrt',
                             max_depth=3, bootstrap =False)
rfc1.fit(X_train, y_train)

 
# Predict on training set
pred_y_1 = rfc1.predict(X_test)
 
# Is our model still predicting just one class?
print( np.unique( pred_y_1 ) )
# [0 1]
 
# How's our accuracy?
print( accuracy_score(y_test, pred_y_1) )


# In[177]:


print(recall_score(y_test, pred_y_1))
confusion_matrix(y_test, pred_y_1)


# In[181]:


import xgboost as xgb
xgb=xgb.XGBClassifier()

xgb.fit(X_train, y_train)

 
# Predict on training set
pred_y_1 = xgb.predict(X_test)
 
# Is our model still predicting just one class?
print( np.unique( pred_y_1 ) )
# [0 1]
 
# How's our accuracy?
print( accuracy_score(y_test, pred_y_1) )
#  from sklearn.XGboost import XGBClassifier


# In[194]:


# pred_y_1


# In[ ]:





# In[182]:


print(recall_score(y_test, pred_y_1))
confusion_matrix(y_test, pred_y_1)


# In[ ]:


# Balabnce bagging


# In[221]:


from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.ensemble import RandomForestClassifier

#Create an object of the classifier.
bbc = BalancedBaggingClassifier(base_estimator=RandomForestClassifier(),
                                 sampling_strategy='auto',
                                replacement=False,
                                random_state=0)

y_train = train['m13']
X_train = train.drop(['m13'], axis = 1)

#Train the classifier.
bbc.fit(X_train, y_train)
pred_y_1 = bbc.predict(X_train)
# print( accuracy_score(y_test, pred_y_1) )
# print(recall_score(y_test, pred_y_1))
# confusion_matrix(y_test, pred_y_1)


# In[3]:


from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#Create an object of the classifier.
bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=100)

# y_train = train['m13']
# X_train = train.drop(['m13'], axis = 1)

#Train the classifier.
bbc.fit(X_train, y_train)
pred_y_1 = bbc.predict(X_train)
# print( accuracy_score(y_test, pred_y_1) )
# print(recall_score(y_test, pred_y_1))
# confusion_matrix(y_test, pred_y_1)


# In[223]:


y_train = train['m13']
X_train = train.drop(['m13'], axis = 1)


for j in [500,2000,8000,99999]:
    clf_stump=DecisionTreeClassifier(max_features=None,max_leaf_nodes=j)
    print(j)
    for i in np.arange(1,max_n_ests):
        baglfy=BaggingClassifier(base_estimator=clf_stump,n_estimators=i,
            max_samples=1.0)
        baglfy=baglfy.fit(X_train,y_train)
        pred_y_1 = baglfy.predict(x)


# In[ ]:





# In[207]:


print( accuracy_score(y_train, pred_y_1) )
print(recall_score(y_train, pred_y_1))
confusion_matrix(y_train, pred_y_1)


# In[222]:


print( accuracy_score(y_train, pred_y_1) )
print(recall_score(y_train, pred_y_1))
confusion_matrix(y_train, pred_y_1)


# In[ ]:





# In[184]:


from sklearn.svm import SVC
svc = SVC(gamma = 0.01, C = 100, random_state = 300, probability=True)
svc.fit(X_train, y_train)
pred_y_1 = xgb.predict(X_test)
 
# Is our model still predicting just one class?
print( np.unique( pred_y_1 ) )
# [0 1]
 
# How's our accuracy?
print( accuracy_score(y_test, pred_y_1) )
print(recall_score(y_test, pred_y_1))
confusion_matrix(y_test, pred_y_1)


# In[ ]:





# In[154]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(3, 10, num = 1)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 2, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, y_train)

print(rf_random.best_params_)


# In[ ]:





# In[ ]:





# In[185]:


#NearMiss--------------------------------------------------------------------------------
from imblearn.under_sampling import NearMiss
X_train, X_test, y_train, y_test = train_test_split(scaled_features, y, random_state = 1, stratify=y)

nr = NearMiss()
X_train, y_train = nr.fit_sample(X_train, y_train)
np.bincount(y_train)


# In[186]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier( n_estimators = 173,min_samples_split=29,min_samples_leaf = 5, max_features = 'sqrt',
                             max_depth=3, bootstrap =True)
rfc.fit(X_train, y_train)

 
# Predict on training set
pred_y_1 = rfc.predict(X_test)
 
# Is our model still predicting just one class?
print( np.unique( pred_y_1 ) )
# [0 1]
 
# How's our accuracy?
print( accuracy_score(y_test, pred_y_1) )


# In[187]:


print(recall_score(y_test, pred_y_1))
confusion_matrix(y_test, pred_y_1)


# In[188]:


import xgboost as xgb
xgb=xgb.XGBClassifier()

xgb.fit(X_train, y_train)

 
# Predict on training set
pred_y_1 = xgb.predict(X_test)
 
# Is our model still predicting just one class?
print( np.unique( pred_y_1 ) )
# [0 1]
 
# How's our accuracy?
print( accuracy_score(y_test, pred_y_1) )
print(recall_score(y_test, pred_y_1))
confusion_matrix(y_test, pred_y_1)


# In[198]:


pred_test =bbc.predict(scaled_features_test)


# In[104]:


# pred_test = pred_test.tolist()
# pred_test


# In[199]:


df2 = pd.DataFrame({'m13':pred_test})


# In[200]:


df2.m13.value_counts()


# In[201]:


submission = pd.DataFrame({ 'loan_id':test['loan_id'],'m13':df2['m13']})
submission.head(3)


# In[202]:


filename = 'Analytics_IMB_data_BBG.csv'

submission.to_csv(filename,index=False)
print('Saved file: ' + filename)

