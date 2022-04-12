#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# # Explore

# In[185]:


df = pd.read_csv('./loans_full_schema.csv')


# ### Obeserve correlation

# In[165]:


plt.figure(figsize=(15,15), dpi=300)
sns.heatmap(df.corr(), square=True)
plt.savefig('./heatmap.png')


# In[48]:


df.describe().T


# In[183]:


print('{} rows have paid total != paid late fees + paid principal + paid interest.'      .format(df[df.paid_interest + df.paid_late_fees + df.paid_principal - df.paid_total > 0.01].shape[0]))


# In[192]:


print('{} rows have paid more or equal to loan amount times interest rate.'      .format(df[df.paid_interest - (df.loan_amount * df.interest_rate * 0.01) > 0.1].shape[0]))


# In[191]:


df.head().T


# It seemed like even for those finished loans, the paid interest some how didn't match the loan amount multiplied to the interest rate.

# ### Turn interest rate into categories for easier visualization

# In[175]:


df.interest_rate = df.interest_rate // 5 * 5
df.interest_rate.replace([5, 10, 15, 20, 25, 30], ['5-10', '10-15', '15-20', '20-25', '25-30', '30+'], inplace=True)
df['interest_rate'] = pd.Categorical(df['interest_rate'], ['5-10', '10-15', '15-20', '20-25', '25-30', '30+'])


# Plot all the fields and see if there was anything special

# In[174]:


df.shape


# In[176]:


fig, axes = plt.subplots(17, 3, figsize=(24, 136))
i = 0
for col in df.columns:
    if(col in ['emp_title', 'state', 'num_accounts_30d_past_due', 'num_accounts_120d_past_due']):
        continue
    g = sns.histplot(data = df, x = col, ax = axes[i//3, i%3], bins=12, hue = 'interest_rate',                     multiple="stack", hue_order=['30+', '25-30', '20-25', '15-20', '10-15', '5-10'])
    g.set(ylabel=None)
    i += 1
plt.savefig('./hist.png')


# ### From the plot above, grade and sub_grade seemed to have high correlation to interest_rate

# In[179]:


df['grade'] = pd.Categorical(df['grade'], sorted(df.grade.unique()))
df['sub_grade'] = pd.Categorical(df['sub_grade'], sorted(df.sub_grade.unique()))


# In[180]:


plt.figure(figsize=(15,5), dpi=300)
sns.histplot(data = df, x = 'grade', bins=12, hue = 'interest_rate',                     multiple="stack", hue_order=['30+', '25-30', '20-25', '15-20', '10-15', '5-10'])
plt.savefig('./grade_hist.png')


# In[181]:


plt.figure(figsize=(15,5), dpi=300)
sns.histplot(data = df, x = 'sub_grade', bins=12, hue = 'interest_rate',                     multiple="stack", hue_order=['30+', '25-30', '20-25', '15-20', '10-15', '5-10'])
plt.savefig('./sub_grade_hist.png')


# # Read and clean data for analysis

# In[2]:


df = pd.read_csv('./loans_full_schema.csv')


# In[194]:


pd.get_dummies(df.grade, prefix='grade')


# In[3]:


df = df.join(pd.get_dummies(df.grade, prefix='grade'))
df = df.join(pd.get_dummies(df.sub_grade, prefix='sub_grade'))


# In[4]:


df.drop(['grade', 'sub_grade'], axis = 1, inplace=True)
df.drop(['num_accounts_120d_past_due', 'num_accounts_30d_past_due', 'current_accounts_delinq', 'paid_total'],        axis=1, inplace=True)


# In[5]:


X = df.select_dtypes(include=np.number).drop('interest_rate', axis = 1)
X.fillna(0, inplace=True)
y = df.interest_rate
y.fillna(0, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[6]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)


# In[7]:


reg.score(X_test, y_test)


# In[9]:


coef_df = pd.DataFrame({'col_name': X.columns, 'coef': reg.coef_}, columns = ['col_name', 'coef'])

coef_df.sort_values(by = 'coef', key = abs, ascending= False).head(10)


# In[17]:


from sklearn import svm
regr = svm.SVR()
regr.fit(X_train, y_train)
regr.score(X_test, y_test)


# In[28]:


df = pd.read_csv('./loans_full_schema.csv')
df['sub_grade'] = pd.Categorical(df['sub_grade'], sorted(df.sub_grade.unique()))
pred_df = df.iloc[X_test.index]
pred_df = pred_df.reset_index(drop = True)
pred_df = pred_df.join(pd.DataFrame(reg.predict(X_test), columns=['predicted_interest_rate']))


# In[11]:


pred_df = pred_df.join(pd.DataFrame(pred_df.interest_rate - pred_df.predicted_interest_rate                                    , columns=['prediction_error']))


# In[15]:


plt.figure(figsize=(15,5), dpi=300)

sns.stripplot(data = pred_df, x = 'sub_grade', y = 'prediction_error', jitter = 0.3, hue = 'grade')
plt.axhline(y=0, color='grey', linestyle='-')
plt.savefig('./linearReg.png')

