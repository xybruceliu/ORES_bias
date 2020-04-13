#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn import metrics
from collections import OrderedDict


# ## Replicate the ORES model

# In[46]:


# read in data
df = pd.read_csv('enwiki.labeled_revisions.20k_2015.csv')
df = df.dropna()


# In[47]:


#defining newcomer as: < 3.637819e+06
df['feature.temporal.revision.user.seconds_since_registration'].describe()


# In[48]:


# Combine anon and new to a 3-category new feature, anonymous, newcomers, experienced
newcomer_seconds = 3.637819e+06

conditions = [
    (df['feature.revision.user.is_anon'] == True),
    (df['feature.revision.user.is_anon'] == False) & (df['feature.temporal.revision.user.seconds_since_registration'] < newcomer_seconds),
    (df['feature.revision.user.is_anon'] == False) & (df['feature.temporal.revision.user.seconds_since_registration'] >= newcomer_seconds)]
choices = [0,1,2]
df['user.type'] = np.select(conditions, choices)
df.head()


# In[49]:


# check how damaging/goodfaith distributed, also by the sensitive feature
# 96% of data is damaging
print(df['damaging'].describe())
print(18572/19319)


# In[51]:


# 3474 many anonymous users
# 13.7% of anonymous users' edits are damaging
print(df[df['user.type'] == 0]['damaging'].describe())
print(1-2995/3474)


# In[52]:


# 1356 many newcoming users
# 14.5% of newcoming users' edits are damaging
print(df[df['user.type'] == 1]['damaging'].describe())
print(1-1159/1356)


# In[53]:


# 14489 many experienced users
# 0.049% of experienced users' edits are damaging
print(df[df['user.type'] == 2]['damaging'].describe())
print(1-14418/14489)


# In[54]:


# add in sample weights
df['sample_weight'] = np.where(df['damaging']==True, 10, 1)


# In[55]:


# delete the two features
df = df.drop(['feature.revision.user.is_anon', 'feature.temporal.revision.user.seconds_since_registration'], axis=1)


# In[56]:


# convert user.type to categorical
df['user.type'] = pd.Categorical(df['user.type'])


# In[57]:


# divide into X, X_weights and y
y = df["damaging"]
X_weights = df.iloc[:,-1].copy()
X = df.iloc[:,4:-1].copy()
X.head()


# In[58]:


# parameters from 
#https://github.com/wikimedia/editquality/blob/master/model_info/enwiki.damaging.md
params= {'min_impurity_decrease': 0.0, 
         'loss': 'deviance', 
         'n_estimators': 700, 
         'min_impurity_split': None, 
         'verbose': 0, 
         'criterion': 'friedman_mse', 
         'subsample': 1.0, 
         #'center': True, 
         #'scale': True, 
         'presort': 'auto', 
         'init': None, 
         #'multilabel': False, 
         'max_depth': 7, 
         'random_state': None, 
         'learning_rate': 0.01, 
         'validation_fraction': 0.1, 
         'warm_start': False, 
         'min_samples_split': 2, 
         'min_samples_leaf': 1, 
         'min_weight_fraction_leaf': 0.0, 
         'n_iter_no_change': None, 
         'max_leaf_nodes': None, 
         'tol': 0.0001, 
         'max_features': 'log2'}
         #'labels': [True, False], 
         #'label_weights': OrderedDict([(True, 10)])


# In[62]:


gb_clf_replicate = GradientBoostingClassifier(**params)
gb_clf_replicate.fit(X, y, sample_weight=X_weights)


# In[141]:


import matplotlib.pyplot as plt

# Feature importance
importances = gb_clf_replicate.feature_importances_
indices = np.argsort(importances)[::-1]
print(indices)
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[148]:


print(X.iloc[:,76])
print(X.iloc[:,42])
print(X.iloc[:,47])
print(X.iloc[:,10])
print(X.iloc[:,11])
print(X.iloc[:,46])
print(X.iloc[:,31])


# In[63]:


from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, precision_score, recall_score

# test on training set
y_pred = gb_clf_replicate.predict(X)

print(accuracy_score(y, y_pred))
print(balanced_accuracy_score(y, y_pred))
print(f1_score(y, y_pred))
print(recall_score(y, y_pred))
print(precision_score(y, y_pred))
print(roc_auc_score(y, y_pred))


# In[64]:


# comparing this to the ORES model
df2 = pd.read_csv('data.csv')
df2 = df2.dropna()
y2_true = df2.damaging
y2_pred = df2.label_damage.astype(bool)

print(accuracy_score(y2_true, y2_pred))
print(balanced_accuracy_score(y2_true, y2_pred))
print(f1_score(y2_true, y2_pred))
print(recall_score(y2_true, y2_pred))
print(precision_score(y2_true, y2_pred))
print(roc_auc_score(y2_true, y2_pred))


# ## Dropping the sensitive feature

# In[65]:


X_no_sen = df.iloc[:,4:-2].copy()
X_no_sen.head()


# In[67]:


gb_clf_no_sen = GradientBoostingClassifier(**params)
gb_clf_no_sen.fit(X_no_sen, y, sample_weight=X_weights)


# In[68]:


# test on training set
y_pred = gb_clf_no_sen.predict(X_no_sen)

print(accuracy_score(y, y_pred))
print(balanced_accuracy_score(y, y_pred))
print(f1_score(y, y_pred))
print(recall_score(y, y_pred))
print(precision_score(y, y_pred))
print(roc_auc_score(y, y_pred))


# In[69]:


# comparing this to the ORES model
print(accuracy_score(y2_true, y2_pred))
print(balanced_accuracy_score(y2_true, y2_pred))
print(f1_score(y2_true, y2_pred))
print(recall_score(y2_true, y2_pred))
print(precision_score(y2_true, y2_pred))
print(roc_auc_score(y2_true, y2_pred))


# ## FairLearn model (with sensitive feature)

# In[70]:


from fairlearn.reductions import GridSearch, ExponentiatedGradient
from fairlearn.reductions import DemographicParity, EqualizedOdds


# In[71]:


# sensitive feature
A = X[['user.type']]
A.head()


# In[127]:


gb_clf_fair=ExponentiatedGradient(GradientBoostingClassifier(**params),
                       constraints=EqualizedOdds(), eps=0.1)

gb_clf_fair.fit(X, y, sensitive_features=A, sample_weight=X_weights)


# In[128]:


# test on training set
y_pred = gb_clf_fair.predict(X)

print(accuracy_score(y, y_pred))
print(balanced_accuracy_score(y, y_pred))
print(f1_score(y, y_pred))
print(recall_score(y, y_pred))
print(precision_score(y, y_pred))
print(roc_auc_score(y, y_pred))


# In[122]:


# comparing this to the ORES model
print(accuracy_score(y2_true, y2_pred))
print(balanced_accuracy_score(y2_true, y2_pred))
print(f1_score(y2_true, y2_pred))
print(recall_score(y2_true, y2_pred))
print(precision_score(y2_true, y2_pred))
print(roc_auc_score(y2_true, y2_pred))


# ## Models performance on test set

# In[75]:


# import test set
df_test = pd.read_csv('ORES_test_data.csv')
df_test.head()


# In[82]:


X_test = df_test.iloc[:,6:].copy()
X_test_no_sen = df_test.iloc[:,6:-1].copy()
y_test = df_test['is_reverted']
X_test.shape


# In[85]:


# ORES model
y_pred = df_test['damaging']
print(accuracy_score(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))


# In[86]:


# replicate model
y_pred = gb_clf_replicate.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))


# In[123]:


# no-sensitive feature model
y_pred = gb_clf_no_sen.predict(X_test_no_sen)

print(accuracy_score(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))


# In[129]:


# fair model
y_pred = gb_clf_fair.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(balanced_accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred))


# ## Biases in these 4 models

# In[89]:


import seaborn as sns
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot


# ### ORES

# In[103]:


# ORES model
df_bias_ORES = pd.DataFrame(columns = ['score', 'label_value', 'user.type'])
df_bias_ORES['label_value'] = y_test
df_bias_ORES['user.type'] = df_test.iloc[:,-1].copy().astype(str)
df_bias_ORES['score'] = df_test['damaging']

g = Group()
xtab, _ = g.get_crosstabs(df_bias_ORES)
absolute_metrics = g.list_absolute_metrics(xtab)
xtab[[col for col in xtab.columns if col not in absolute_metrics]]


# In[104]:


xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(4)


# ### Replicate model

# In[105]:


# Replicate model
df_bias_rep = pd.DataFrame(columns = ['score', 'label_value', 'user.type'])
df_bias_rep['label_value'] = y_test
df_bias_rep['user.type'] = df_test.iloc[:,-1].copy().astype(str)
df_bias_rep['score'] = gb_clf_replicate.predict(X_test)

g = Group()
xtab, _ = g.get_crosstabs(df_bias_rep)
absolute_metrics = g.list_absolute_metrics(xtab)
xtab[[col for col in xtab.columns if col not in absolute_metrics]]


# In[106]:


xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(4)


# ### No-sensitive Feature Model

# In[107]:


# No sensitive feature model
df_bias_nosen = pd.DataFrame(columns = ['score', 'label_value', 'user.type'])
df_bias_nosen['label_value'] = y_test
df_bias_nosen['user.type'] = df_test.iloc[:,-1].copy().astype(str)
df_bias_nosen['score'] = gb_clf_no_sen.predict(X_test_no_sen)

g = Group()
xtab, _ = g.get_crosstabs(df_bias_nosen)
absolute_metrics = g.list_absolute_metrics(xtab)
xtab[[col for col in xtab.columns if col not in absolute_metrics]]


# In[108]:


xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(4)


# ### Fair Model

# In[130]:


# Fair model
df_bias_fair = pd.DataFrame(columns = ['score', 'label_value', 'user.type'])
df_bias_fair['label_value'] = y_test
df_bias_fair['user.type'] = df_test.iloc[:,-1].copy().astype(str)
df_bias_fair['score'] = gb_clf_fair.predict(X_test)

g = Group()
xtab, _ = g.get_crosstabs(df_bias_fair)
absolute_metrics = g.list_absolute_metrics(xtab)
xtab[[col for col in xtab.columns if col not in absolute_metrics]]


# In[131]:


xtab[['attribute_name', 'attribute_value'] + absolute_metrics].round(4)


# In[ ]:




