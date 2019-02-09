
# coding: utf-8

# In[1]:


# LOADING MODULES

import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

#pd.options.mode.chained_assignment = None
#pd.options.display.max_columns = 999

import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
get_ipython().run_line_magic('matplotlib', 'inline')

from pydoc import help
from scipy.stats.stats import pearsonr

from datetime import datetime
from datetime import timedelta
import datetime as dt
import calendar

import math


# In[ ]:


#DELETE
train_df = pd.read_csv("./processed_train.csv")
test_df = pd.read_csv("./processed_test.csv")


# In[ ]:


#DELETE
print (train_df.columns)
print (train_df.dtypes)


# In[ ]:


# EXTRACTING FEATURES FROM THE WEATHER DATASET

external=pd.read_csv("./external_data2.csv")


# In[ ]:


print (external.columns)
external.columns = ['Index', 'geoNetwork_country', 'InternetUsers', 'Population', 'NonUsers']
print (external.columns)


# In[ ]:


# MERGING EXISTING DATA WITH WEATHER DATA TO GENERATE NEW FEATURES

train_new = pd.merge(train_df, external, how='left', on=['geoNetwork_country'])

print (train_new.columns)


# In[ ]:


print (train_new.shape)
print (train_new.columns)
print (train_new.dtypes)


# In[ ]:


print (train_new.geoNetwork_country.unique())
#print (train_new.geoNetwork_country.size)


# In[ ]:


print (train_df.geoNetwork_country.unique())


# In[74]:


# LOAD DATA

initial_train = pd.read_csv("./train.csv")
initial_test = pd.read_csv("./test.csv")


# In[ ]:


print (initial_train.shape)
print (initial_test.shape)

print (initial_train.columns)
print (initial_test.columns)

print (initial_train.dtypes)


# In[2]:


# FUNCTION TO EXTRACT JSON VALUES

def load_df(filename):
    csv_path = "./" + filename
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}
                    )
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df


# In[3]:


train_df = load_df("train.csv")
test_df = load_df("test.csv")


# In[ ]:


train_full_df = train_df
test_ful_df = test_df


# In[ ]:


print ("TRAIN")
print (train_df.shape)
print (train_df.columns)
print (train_df.dtypes)

print ("TEST")
print (test_df.shape)
print (test_df.columns)
print (test_df.dtypes)


# In[ ]:


train_df.device_browser.unique()


# In[ ]:


train_df.device_browserSize.unique()


# In[ ]:


train_df.device_browserVersion.unique()


# In[ ]:


train_df.device_deviceCategory.unique()


# In[ ]:


train_df.device_flashVersion.unique()


# In[ ]:


train_df.device_isMobile.unique()


# In[ ]:


train_df.device_language.unique()


# In[ ]:


train_df.device_mobileDeviceBranding.unique()


# In[ ]:


train_df.device_mobileDeviceInfo.unique()


# In[ ]:


train_df.device_mobileDeviceMarketingName.unique()


# In[ ]:


train_df.device_mobileDeviceModel.unique()


# In[ ]:


train_df.device_mobileInputSelector.unique()


# In[ ]:


train_df.device_operatingSystem.unique()


# In[ ]:


train_df.device_operatingSystemVersion.unique()


# In[ ]:


train_df.device_screenColors.unique()


# In[ ]:


train_df.device_screenResolution.unique()


# In[ ]:


train_df.geoNetwork_city.unique()


# In[ ]:


train_df.geoNetwork_cityId.unique()


# In[ ]:


train_df.geoNetwork_continent.unique()


# In[ ]:


train_df.geoNetwork_country.unique()


# In[ ]:


train_df.geoNetwork_latitude.unique()


# In[ ]:


train_df.geoNetwork_longitude.unique()


# In[ ]:


train_df.geoNetwork_metro.unique()


# In[ ]:


train_df.geoNetwork_networkDomain.unique()


# In[ ]:


train_df.geoNetwork_networkLocation.unique()


# In[ ]:


train_df.geoNetwork_region.unique()


# In[ ]:


train_df.geoNetwork_subContinent.unique()


# In[ ]:


train_df.totals_bounces.unique()


# In[ ]:


train_df.totals_hits.unique()


# In[ ]:


train_df.totals_newVisits.unique()


# In[ ]:


train_df.totals_pageviews.unique()


# In[ ]:


train_df.totals_visits.unique()


# In[ ]:


train_df.trafficSource_adContent.unique()


# In[ ]:


train_df.trafficSource_adwordsClickInfo.adNetworkType.unique()


# In[ ]:


train_df.trafficSource_adwordsClickInfo_criteriaParameters.unique()


# In[ ]:


train_df.trafficSource_adwordsClickInfo_gclId.unique()


# In[ ]:


train_df.trafficSource_adwordsClickInfo_isVideoAd.unique()


# In[ ]:


train_df.trafficSource_adwordsClickInfo_page.unique()


# In[ ]:


train_df.trafficSource_adwordsClickInfo_slot.unique()


# In[ ]:


train_df.trafficSource_campaign.unique()


# In[ ]:


train_df.trafficSource_isTrueDirect.unique()


# In[ ]:


train_df.trafficSource_keyword.unique()


# In[ ]:


train_df.trafficSource_medium.unique()


# In[ ]:


train_df.trafficSource_referralPath.unique()


# In[ ]:


train_df.trafficSource_source.unique()


# In[4]:


# CONVERTING TRANSACTION REVENUE TO FLOAT

train_df["totals_transactionRevenue"] = train_df["totals_transactionRevenue"].astype('float')


# In[ ]:


nzi = pd.notnull(train_df["totals_transactionRevenue"]).sum()
nzr = (gdf["totals_transactionRevenue"]>0).sum()
print("Number of instances in train set with non-zero revenue : ", nzi, " and ratio is : ", nzi / train_df.shape[0])
print("Number of unique customers with non-zero revenue : ", nzr, "and the ratio is : ", nzr / gdf.shape[0])


# In[ ]:


print("Number of unique visitors in train set : ",train_df.fullVisitorId.nunique(), " out of rows : ",train_df.shape[0])
print("Number of unique visitors in test set : ",test_df.fullVisitorId.nunique(), " out of rows : ",test_df.shape[0])
print("Number of common visitors in train and test set : ",len(set(train_df.fullVisitorId.unique()).intersection(set(test_df.fullVisitorId.unique())) ))


# In[ ]:


#VERIFY
gdf = train_df.groupby("fullVisitorId")["totals_transactionRevenue"].sum().reset_index()
print (gdf.columns)

plt.figure(figsize=(8,6))
plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totals_transactionRevenue"].values)))
plt.xlabel('index', fontsize=12)
plt.ylabel('TransactionRevenue', fontsize=12)
plt.show()


# In[ ]:


#VERIFY
plt.figure(figsize=(8,6))
sns.distplot(np.log1p(non_zero))
plt.title("Log Distribution of Non Zero Total Transactions");
plt.xlabel("Log - Total Transactions");


# In[ ]:


# DEVICE ATTRIBUTES


# In[13]:


fig1 = tools.make_subplots(rows=1, cols=2, subplot_titles=["Count: Device Category", "Non-zero Revenue Count: Device Category"], print_grid=False)

trs1 = []

tmp1 = train_df.groupby('device_deviceCategory').agg({"totals_transactionRevenue": "size"}).reset_index().rename(columns={"totals_transactionRevenue" : "Count"})
tmp1 = tmp1.dropna().sort_values("Count", ascending = False)
tr1 = go.Bar(y = tmp1["Count"], orientation="v", marker=dict(opacity=0.5, color="blue"), x= tmp1['device_deviceCategory'])
trs1.append(tr1)

tmp1 = train_df.groupby('device_deviceCategory').agg({"totals_transactionRevenue": "count"}).reset_index().rename(columns={"totals_transactionRevenue" : "Non-zero Revenue Count"})
tmp1 = tmp1.dropna().sort_values("Non-zero Revenue Count", ascending = False)
tr1 = go.Bar(y = tmp1["Non-zero Revenue Count"], orientation="v", marker=dict(opacity=0.5, color="blue"), x= tmp1['device_deviceCategory'])
trs1.append(tr1)

fig1.append_trace(trs1[0], 1, 1)
fig1.append_trace(trs1[1], 1, 2)
fig1['layout'].update(height=450, margin=dict(b=200), showlegend=False)
py.iplot(fig1)

fig2 = tools.make_subplots(rows=1, cols=2, subplot_titles=["Count: Device OperatingSystem", "Non-zero Revenue Count: Device OperatingSystem"], print_grid=False)

trs2 = []

tmp2 = train_df.groupby('device_operatingSystem').agg({"totals_transactionRevenue": "size"}).reset_index().rename(columns={"totals_transactionRevenue" : "Count"})
tmp2 = tmp2.dropna().sort_values("Count", ascending = False)
tr2 = go.Bar(y = tmp2["Count"], orientation="v", marker=dict(opacity=0.5, color="orange"), x= tmp2['device_operatingSystem'])
trs2.append(tr2)

tmp2 = train_df.groupby('device_operatingSystem').agg({"totals_transactionRevenue": "count"}).reset_index().rename(columns={"totals_transactionRevenue" : "Non-zero Revenue Count"})
tmp2 = tmp2.dropna().sort_values("Non-zero Revenue Count", ascending = False)
tr2 = go.Bar(y = tmp2["Non-zero Revenue Count"], orientation="v", marker=dict(opacity=0.5, color="orange"), x= tmp2['device_operatingSystem'])
trs2.append(tr2)

fig2.append_trace(trs2[0], 1, 1)
fig2.append_trace(trs2[1], 1, 2)
fig2['layout'].update(height=450, margin=dict(b=200), showlegend=False)
py.iplot(fig2)


# In[11]:


#GEOGRAPHY ATTRIBUTES

fig1 = tools.make_subplots(rows=1, cols=2, subplot_titles=["Count: Continent", "Non-zero Revenue Count: Continent"], print_grid=False)

trs1 = []

tmp1 = train_df.groupby('geoNetwork_continent').agg({"totals_transactionRevenue": "size"}).reset_index().rename(columns={"totals_transactionRevenue" : "Count"})
tmp1 = tmp1.dropna().sort_values("Count", ascending = False)
tr1 = go.Bar(y = tmp1["Count"], orientation="v", marker=dict(opacity=0.5, color="blue"), x= tmp1['geoNetwork_continent'])
trs1.append(tr1)

tmp1 = train_df.groupby('geoNetwork_continent').agg({"totals_transactionRevenue": "count"}).reset_index().rename(columns={"totals_transactionRevenue" : "Non-zero Revenue Count"})
tmp1 = tmp1.dropna().sort_values("Non-zero Revenue Count", ascending = False)
tr1 = go.Bar(y = tmp1["Non-zero Revenue Count"], orientation="v", marker=dict(opacity=0.5, color="blue"), x= tmp1['geoNetwork_continent'])
trs1.append(tr1)

fig1.append_trace(trs1[0], 1, 1)
fig1.append_trace(trs1[1], 1, 2)
fig1['layout'].update(height=450, margin=dict(b=200), showlegend=False)
py.iplot(fig1)

fig2 = tools.make_subplots(rows=1, cols=2, subplot_titles=["Count: SubContinent", "Non-zero Revenue Count: SubContinent"], print_grid=False)

trs2 = []

tmp2 = train_df.groupby('geoNetwork_subContinent').agg({"totals_transactionRevenue": "size"}).reset_index().rename(columns={"totals_transactionRevenue" : "Count"})
tmp2 = tmp2.dropna().sort_values("Count", ascending = False)
tr2 = go.Bar(y = tmp2["Count"], orientation="v", marker=dict(opacity=0.5, color="orange"), x= tmp2['geoNetwork_subContinent'])
trs2.append(tr2)

tmp2 = train_df.groupby('geoNetwork_subContinent').agg({"totals_transactionRevenue": "count"}).reset_index().rename(columns={"totals_transactionRevenue" : "Non-zero Revenue Count"})
tmp2 = tmp2.dropna().sort_values("Non-zero Revenue Count", ascending = False)
tr2 = go.Bar(y = tmp2["Non-zero Revenue Count"], orientation="v", marker=dict(opacity=0.5, color="orange"), x= tmp2['geoNetwork_subContinent'])
trs2.append(tr2)

fig2.append_trace(trs2[0], 1, 1)
fig2.append_trace(trs2[1], 1, 2)
fig2['layout'].update(height=450, margin=dict(b=200), showlegend=False)
py.iplot(fig2)


# In[16]:


def date_parser(df):
    df['date'] = pd.to_datetime(df['date'].astype(str))
    df['visitStartTime'] = pd.to_datetime(df['visitStartTime'], unit = 's')
    return df

train_df = date_parser(train_df)
test_df = date_parser(test_df)

def add_time_features(df):
    df['dayofweek'] = df['visitStartTime'].dt.dayofweek
    df['hour'] = df['visitStartTime'].dt.hour
    df['month'] = df['visitStartTime'].dt.month
    df[['dayofweek','hour','month']] = df[['dayofweek','hour','month']].apply(lambda x : x.astype('float') )
    return df

train_df = add_time_features(train_df)
test_df = add_time_features(test_df)


# In[23]:


#DATE ATTRIBUTES

fig1 = tools.make_subplots(rows=1, cols=2, subplot_titles=["Count: Hour", "Non-zero Revenue Count: Hour"], print_grid=False)

trs1 = []

tmp1 = train_df.groupby('hour').agg({"totals_transactionRevenue": "size"}).reset_index().rename(columns={"totals_transactionRevenue" : "Count"})
tmp1 = tmp1.dropna().sort_values("Count", ascending = False)
tr1 = go.Bar(y = tmp1["Count"], orientation="v", marker=dict(opacity=0.5, color="blue"), x= tmp1['hour'])
trs1.append(tr1)

tmp1 = train_df.groupby('hour').agg({"totals_transactionRevenue": "count"}).reset_index().rename(columns={"totals_transactionRevenue" : "Non-zero Revenue Count"})
tmp1 = tmp1.dropna().sort_values("Non-zero Revenue Count", ascending = False)
tr1 = go.Bar(y = tmp1["Non-zero Revenue Count"], orientation="v", marker=dict(opacity=0.5, color="blue"), x= tmp1['hour'])
trs1.append(tr1)

fig1.append_trace(trs1[0], 1, 1)
fig1.append_trace(trs1[1], 1, 2)
fig1['layout'].update(height=450, margin=dict(b=200), showlegend=False)
py.iplot(fig1)

fig2 = tools.make_subplots(rows=1, cols=2, subplot_titles=["Count: Day of Week", "Non-zero Revenue Count: Day of Week"], print_grid=False)

trs2 = []

tmp2 = train_df.groupby('dayofweek').agg({"totals_transactionRevenue": "size"}).reset_index().rename(columns={"totals_transactionRevenue" : "Count"})
tmp2 = tmp2.dropna().sort_values("Count", ascending = False)
tr2 = go.Bar(y = tmp2["Count"], orientation="v", marker=dict(opacity=0.5, color="orange"), x= tmp2['dayofweek'])
trs2.append(tr2)

tmp2 = train_df.groupby('dayofweek').agg({"totals_transactionRevenue": "count"}).reset_index().rename(columns={"totals_transactionRevenue" : "Non-zero Revenue Count"})
tmp2 = tmp2.dropna().sort_values("Non-zero Revenue Count", ascending = False)
tr2 = go.Bar(y = tmp2["Non-zero Revenue Count"], orientation="v", marker=dict(opacity=0.5, color="orange"), x= tmp2['dayofweek'])
trs2.append(tr2)

fig2.append_trace(trs2[0], 1, 1)
fig2.append_trace(trs2[1], 1, 2)
fig2['layout'].update(height=450, margin=dict(b=200), showlegend=False)
py.iplot(fig2)

fig3 = tools.make_subplots(rows=1, cols=2, subplot_titles=["Count: Month", "Non-zero Revenue Count: Month"], print_grid=False)

trs3 = []

tmp3 = train_df.groupby('month').agg({"totals_transactionRevenue": "size"}).reset_index().rename(columns={"totals_transactionRevenue" : "Count"})
tmp3 = tmp3.dropna().sort_values("Count", ascending = False)
tr3 = go.Bar(y = tmp3["Count"], orientation="v", marker=dict(opacity=0.5, color="cyan"), x= tmp3['month'])
trs3.append(tr3)

tmp3 = train_df.groupby('month').agg({"totals_transactionRevenue": "count"}).reset_index().rename(columns={"totals_transactionRevenue" : "Non-zero Revenue Count"})
tmp3 = tmp3.dropna().sort_values("Non-zero Revenue Count", ascending = False)
tr3 = go.Bar(y = tmp3["Non-zero Revenue Count"], orientation="v", marker=dict(opacity=0.5, color="cyan"), x= tmp3['month'])
trs3.append(tr3)

fig3.append_trace(trs3[0], 1, 1)
fig3.append_trace(trs3[1], 1, 2)
fig3['layout'].update(height=450, margin=dict(b=200), showlegend=False)
py.iplot(fig3)


# In[69]:


tmp = train_df["channelGrouping"].value_counts()
colors = ["#8d44fc", "#ed95d5", "#caadf7", "#6161b7", "#7e7eba", "#babad1"]
trace = go.Pie(labels=tmp.index, values=tmp.values, marker=dict(colors=colors))
layout = go.Layout(title="Channel Grouping", height=400)
fig = go.Figure(data = [trace], layout = layout)
py.iplot(fig, filename='basic_pie_chart')


# In[24]:


# Probability

cnt_srs1 = train_df.groupby('fullVisitorId')['totals_transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs1.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs1 = cnt_srs1.sort_values(by="count", ascending=False)


# In[25]:


print (cnt_srs1.shape)
print (cnt_srs1.min())
print (cnt_srs1.max())


# In[26]:


prob1 = cnt_srs1
prob1.shape


# In[27]:


prob1["probability"] = cnt_srs1['count of non-zero revenue']/cnt_srs1['count']
#prob["count"] = cnt_srs['count']
#prob["count of non-zero revenue"] = cnt_srs['count of non-zero revenue']


# In[28]:


print (prob1.shape)
print (prob1.columns)
print (prob1.head(10))


# In[29]:


prob1 = prob1.sort_values(by="probability", ascending=False)


# In[72]:


prob_high = prob1[prob1['count'] > 6]


# In[73]:


print (prob_high.shape)
print (prob_high.columns)
print (prob_high.head(10))


# In[32]:


print (prob1.shape)
print (prob1.columns)
print (prob1.head(10))


# In[ ]:


actual_prob = train_df.groupby('fullVisitorId')['totals_transactionRevenue'].agg(['size', 'count'])


# In[ ]:


print (actual_prob.shape)
print (actual_prob.columns)
print (actual_prob.head(10))


# In[ ]:


pref_df = pd.DataFrame({"fullVisitorId":train_df["fullVisitorId"].values})
pred_df['transactionRevenue'] = train_df['totals_transactionRevenue'].values
pred_df['count'] = actual_prob['count'] 


# In[ ]:


print (pref_df.shape)
print (pref_df.columns)
print (pref_df.head(10))


# In[ ]:


train_prob["probability"] = cnt_srs['count of non-zero revenue']/cnt_srs['count']
train_prob["count"] = cnt_srs['count']
train_prob["count of non-zero revenue"] = cnt_srs['count of non-zero revenue']


# In[ ]:


train_prob.shape


# In[ ]:


train_prob.columns = ["probability", "count of non-zero revenue", "mean"]


# In[33]:


print("Variables not in test but in train : ", set(train_df.columns).difference(set(test_df.columns)))


# In[35]:


const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False)==1 ]
const_cols


# In[36]:


cols_to_drop = const_cols + ['sessionId']

train_df = train_df.drop(cols_to_drop + ["trafficSource_campaignCode"], axis=1)
test_df = test_df.drop(cols_to_drop, axis=1)


# In[37]:


print ("TRAIN")
print (train_df.shape)
print (train_df.columns)
print (train_df.dtypes)

print ("TEST")
print (test_df.shape)
print (test_df.columns)
print (test_df.dtypes)


# In[39]:


#MAPPING STRING VALUES TO INTEGERS
from sklearn import model_selection, preprocessing, metrics

# Impute 0 for missing target values
train_df["totals_transactionRevenue"].fillna(0, inplace=True)
train_y = train_df["totals_transactionRevenue"].values
train_id = train_df["fullVisitorId"].values
test_id = test_df["fullVisitorId"].values


# label encode the categorical variables and convert the numerical variables to float
cat_cols = ["channelGrouping", "device_browser", 
            "device_deviceCategory", "device_operatingSystem", 
            "geoNetwork_city", "geoNetwork_continent", 
            "geoNetwork_country", "geoNetwork_metro",
            "geoNetwork_networkDomain", "geoNetwork_region", 
            "geoNetwork_subContinent", "trafficSource_adContent", 
            "trafficSource_adwordsClickInfo.adNetworkType", 
            "trafficSource_adwordsClickInfo.gclId", 
            "trafficSource_adwordsClickInfo.page", 
            "trafficSource_adwordsClickInfo.slot", "trafficSource_campaign",
            "trafficSource_keyword", "trafficSource_medium", 
            "trafficSource_referralPath", "trafficSource_source",
            'trafficSource_adwordsClickInfo.isVideoAd', 'trafficSource_isTrueDirect']

for col in cat_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))


# In[ ]:


#EXTERNAL DATA FROM STRINGS TO INTEGERS
from sklearn import model_selection, preprocessing, metrics

# Impute 0 for missing target values
train_new["totals_transactionRevenue"].fillna(0, inplace=True)
train_y = train_new["totals_transactionRevenue"].values
train_id = train_new["fullVisitorId"].values
test_id = test_df["fullVisitorId"].values


# label encode the categorical variables and convert the numerical variables to float
cat_cols = ["channelGrouping", "device_browser", 
            "device_deviceCategory", "device_operatingSystem", 
            "geoNetwork_city", "geoNetwork_continent", 
            "geoNetwork_country", "geoNetwork_metro",
            "geoNetwork_networkDomain", "geoNetwork_region", 
            "geoNetwork_subContinent", "trafficSource_adContent", 
            "trafficSource_adwordsClickInfo.adNetworkType", 
            "trafficSource_adwordsClickInfo.gclId", 
            "trafficSource_adwordsClickInfo.page", 
            "trafficSource_adwordsClickInfo.slot", "trafficSource_campaign",
            "trafficSource_keyword", "trafficSource_medium", 
            "trafficSource_referralPath", "trafficSource_source",
            'trafficSource_adwordsClickInfo.isVideoAd', 'trafficSource_isTrueDirect']

for col in cat_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_new[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_new[col] = lbl.transform(list(train_new[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))


# In[46]:


#CONVERTING OBJECT TO FLOAT
print (train_df.columns)
num_cols = ["totals_hits", "totals_pageviews", "visitNumber", "totals_bounces",  "totals_newVisits", "dayofweek", "hour", "month"]    

for col in num_cols:
    train_df[col] = train_df[col].astype(float)
    test_df[col] = test_df[col].astype(float)


# In[ ]:


#SPLITTING DATA BASED ON DATE
import datetime

# Split the train dataset into development and valid based on time 
dev_df = train_df[train_df['date']<=datetime.date(2017,5,31)]
val_df = train_df[train_df['date']>datetime.date(2017,5,31)]
dev_y = np.log1p(dev_df["totals_transactionRevenue"].values)
val_y = np.log1p(val_df["totals_transactionRevenue"].values)

dev_X = dev_df[cat_cols + num_cols] 
val_X = val_df[cat_cols + num_cols] 
test_X = test_df[cat_cols + num_cols] 


# In[ ]:


print (dev_df.shape)
print (dev_df.columns)

print (val_df.shape)
print (val_df.columns)

print (dev_y.shape)

print (val_y.shape)

print (dev_X.shape)
print (dev_X.columns)

print (val_X.shape)
print (val_X.columns)

print (test_X.shape)
print (test_X.columns)


# In[47]:


#SPLITTING DATA AS 75% FOR TRAIN AND 25% FOR VALIDATION 

x, y = train_df.shape
x1 = 3 * int(x / 4)

dev_df = train_df[0:x1]
val_df = train_df[x1:x]

dev_X = dev_df
val_X = val_df

dev_y = np.log1p(dev_X["totals_transactionRevenue"].values)
val_y = np.log1p(val_X["totals_transactionRevenue"].values)

dev_X = dev_X.drop(['totals_transactionRevenue'] + ['date'] + ['fullVisitorId'] + ['visitStartTime'], axis=1)
val_X = val_X.drop(['totals_transactionRevenue'] + ['date'] + ['fullVisitorId'] + ['visitStartTime'], axis=1)

test_X = test_df
test_X = test_X.drop(['date'] + ['fullVisitorId'] + ['visitStartTime'], axis=1)

print (dev_y.shape)

print (val_y.shape)

print (dev_X.shape)
print (dev_X.columns)

print (val_X.shape)
print (val_X.columns)

print (test_X.shape)
print (test_X.columns)


# In[ ]:


# import sys
# !conda install --yes --prefix {sys.prefix} lightgbm


# In[ ]:


# !conda install --yes lightgbm


# In[ ]:


# !conda install --yes -c conda-forge lightgbm


# In[48]:


print ("TRAIN")
print (train_df.shape)
print (train_df.columns)

print ("TEST")
print (test_df.shape)
print (test_df.columns)


# In[ ]:


imp_cols = ["channelGrouping", 
            "device_operatingSystem", 
            "geoNetwork_city", "geoNetwork_region",
            "geoNetwork_country", 
            "geoNetwork_metro",
            "geoNetwork_networkDomain", 
            "trafficSource_adwordsClickInfo.gclId", 
            "trafficSource_referralPath", "trafficSource_source",
            "totals_hits", "totals_pageviews", "visitNumber"
            , "visitId"
           ]

dev_X = dev_X[imp_cols] 
val_X = val_X[imp_cols] 
test_X = test_X[imp_cols] 


# In[ ]:


import lightgbm as lgb


# In[49]:


train_df.dtypes


# In[50]:


# LGB
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "num_leaves" : 30,
        "min_child_samples" : 100,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    return pred_test_y, model, pred_val_y

# Training the model #
pred_test, model, pred_val = run_lgb(dev_X, dev_y, val_X, val_y, test_X)


# In[51]:


from sklearn import metrics
pred_val[pred_val<0] = 0
val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
val_pred_df["transactionRevenue"] = val_df["totals_transactionRevenue"].values
val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
#print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))


# In[52]:


sub_df = pd.DataFrame({"fullVisitorId":test_id})
pred_test[pred_test<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("prediction_1.csv", index=False)


# In[53]:


fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()


# In[55]:


# LGB FOR PERMUTATION TEST
def run_lgb_perm(train_X, train_y, val_X, val_y):
    params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "num_leaves" : 30,
        "min_child_samples" : 100,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
    
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    return model, pred_val_y


# In[56]:


base_rmse = 1.60683
iter = 100
p_val = []

#for column_perm in ["totals_pageviews", "totals_hits", "visitID", "totals_newVisits", "geoNetwork_continent", "trafficSource_campaign"]:
for column_perm in ["totals_pageviews", "totals_hits", "trafficSource_campaign"]:
    ans = []
    for i in range(iter):
        print(column_perm," ",i)
        train_permute = train_df.copy()
        train_permute[column_perm] = np.random.permutation(train_permute[column_perm])
        x, y = train_df.shape
        x1 = 3 * int(x / 4)

        dev_df = train_permute[0:x1]
        val_df = train_permute[x1:x]

        dev_X = dev_df
        val_X = val_df

        dev_y = np.log1p(dev_X["totals_transactionRevenue"].values)
        val_y = np.log1p(val_X["totals_transactionRevenue"].values)

        dev_X = dev_X.drop(['totals_transactionRevenue'] + ['date'] + ['fullVisitorId'] + ['visitStartTime'], axis=1)
        val_X = val_X.drop(['totals_transactionRevenue'] + ['date'] + ['fullVisitorId'] + ['visitStartTime'], axis=1)

        model, pred_val = run_lgb_perm(dev_X, dev_y, val_X, val_y)
        
        from sklearn import metrics
        pred_val[pred_val<0] = 0
        val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
        val_pred_df["transactionRevenue"] = val_df["totals_transactionRevenue"].values
        val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
        val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
        print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
        ans.append(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
    
    less_sum = sum(i <= base_rmse for i in ans)
    p_val.append(less_sum/iter)


# In[60]:


print (p_val)


# In[58]:


base_rmse = 1.6288250113593938
iter = 100
p_val = []

#for column_perm in ["totals_pageviews", "totals_hits", "visitID", "totals_newVisits", "geoNetwork_continent", "trafficSource_campaign"]:
for column_perm in ["totals_pageviews", "trafficSource_campaign"]:
    ans = []
    for i in range(iter):
        print(column_perm," ",i)
        train_permute = train_df.copy()
        train_permute[column_perm] = np.random.permutation(train_permute[column_perm])
        x, y = train_df.shape
        x1 = 3 * int(x / 4)

        dev_df = train_permute[0:x1]
        val_df = train_permute[x1:x]

        dev_X = dev_df
        val_X = val_df

        dev_y = np.log1p(dev_X["totals_transactionRevenue"].values)
        val_y = np.log1p(val_X["totals_transactionRevenue"].values)

        dev_X = dev_X.drop(['totals_transactionRevenue'] + ['date'] + ['fullVisitorId'] + ['visitStartTime'], axis=1)
        val_X = val_X.drop(['totals_transactionRevenue'] + ['date'] + ['fullVisitorId'] + ['visitStartTime'], axis=1)

        model, pred_val = run_lgb_perm(dev_X, dev_y, val_X, val_y)
        
        from sklearn import metrics
        pred_val[pred_val<0] = 0
        val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
        val_pred_df["transactionRevenue"] = val_df["totals_transactionRevenue"].values
        val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
        val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
        print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
        ans.append(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
    
    less_sum = sum(i <= base_rmse for i in ans)
    p_val.append(less_sum/iter)


# In[63]:


print ("pval for totals_pageviews and trafficSource_campaign")
pval1 = p_val
print (pval1)


# In[64]:


base_rmse = 1.6288250113593938
iter = 100
p_val = []

for column_perm in ["totals_hits", "visitID", "totals_newVisits", "geoNetwork_continent", "trafficSource_adwordsClickInfo.isVideoAd"]:
#for column_perm in ["totals_pageviews", "trafficSource_campaign"]:
    ans = []
    for i in range(iter):
        print(column_perm," ",i)
        train_permute = train_df.copy()
        train_permute[column_perm] = np.random.permutation(train_permute[column_perm])
        x, y = train_df.shape
        x1 = 3 * int(x / 4)

        dev_df = train_permute[0:x1]
        val_df = train_permute[x1:x]

        dev_X = dev_df
        val_X = val_df

        dev_y = np.log1p(dev_X["totals_transactionRevenue"].values)
        val_y = np.log1p(val_X["totals_transactionRevenue"].values)

        dev_X = dev_X.drop(['totals_transactionRevenue'] + ['date'] + ['fullVisitorId'] + ['visitStartTime'], axis=1)
        val_X = val_X.drop(['totals_transactionRevenue'] + ['date'] + ['fullVisitorId'] + ['visitStartTime'], axis=1)

        model, pred_val = run_lgb_perm(dev_X, dev_y, val_X, val_y)
        
        from sklearn import metrics
        pred_val[pred_val<0] = 0
        val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
        val_pred_df["transactionRevenue"] = val_df["totals_transactionRevenue"].values
        val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
        val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
        print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
        ans.append(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
    
    less_sum = sum(i <= base_rmse for i in ans)
    p_val.append(less_sum/iter)


# In[65]:


print ("pval for totals_hits")
pval2 = p_val
print (pval2)


# In[66]:


base_rmse = 1.6288250113593938
iter = 100
p_val = []

for column_perm in ["visitId", "totals_newVisits", "geoNetwork_continent", "trafficSource_adwordsClickInfo.isVideoAd"]:
#for column_perm in ["totals_pageviews", "trafficSource_campaign"]:
    ans = []
    for i in range(iter):
        print(column_perm," ",i)
        train_permute = train_df.copy()
        train_permute[column_perm] = np.random.permutation(train_permute[column_perm])
        x, y = train_df.shape
        x1 = 3 * int(x / 4)

        dev_df = train_permute[0:x1]
        val_df = train_permute[x1:x]

        dev_X = dev_df
        val_X = val_df

        dev_y = np.log1p(dev_X["totals_transactionRevenue"].values)
        val_y = np.log1p(val_X["totals_transactionRevenue"].values)

        dev_X = dev_X.drop(['totals_transactionRevenue'] + ['date'] + ['fullVisitorId'] + ['visitStartTime'], axis=1)
        val_X = val_X.drop(['totals_transactionRevenue'] + ['date'] + ['fullVisitorId'] + ['visitStartTime'], axis=1)

        model, pred_val = run_lgb_perm(dev_X, dev_y, val_X, val_y)
        
        from sklearn import metrics
        pred_val[pred_val<0] = 0
        val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
        val_pred_df["transactionRevenue"] = val_df["totals_transactionRevenue"].values
        val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
        val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
        print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
        ans.append(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
    
    less_sum = sum(i <= base_rmse for i in ans)
    p_val.append(less_sum/iter)


# In[67]:


print ("pval for visitId, totals_newVisits, geoNetwork_continent, trafficSource_adwordsClickInfo.isVideoAd")
pval3 = p_val
print (pval3)


# In[ ]:


tmp_sum = sum(i <= 1.60683 for i in ans)
print (tmp_sum)


# In[ ]:


ans


# In[ ]:


#IGNORE
for column_perm in ["totals_pageviews", "totals_hits", "visitNumber", "geoNetwork_country", "totals_pageviews"]:
    for i in range(20):
        print(column_perm," ",i)
        train_permute = train_df.copy()
        train_permute[column_perm] = np.random.permutation(train_permute[column_perm])
        x, y = train_df.shape
        x1 = 3 * int(x / 4)

        dev_df = train_permute[0:x1]
        val_df = train_permute[x1:x]

        dev_X = dev_df
        val_X = val_df

        dev_y = np.log1p(dev_X["totals_transactionRevenue"].values)
        val_y = np.log1p(val_X["totals_transactionRevenue"].values)

        dev_X = dev_X.drop(['totals_transactionRevenue'] + ['date'] + ['fullVisitorId'] + ['visitStartTime'], axis=1)
        val_X = val_X.drop(['totals_transactionRevenue'] + ['date'] + ['fullVisitorId'] + ['visitStartTime'], axis=1)

        model, pred_val = run_lgb_perm(dev_X, dev_y, val_X, val_y)
        
        #new_test1.csv
        from sklearn import metrics
        pred_val[pred_val<0] = 0
        val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
        val_pred_df["transactionRevenue"] = val_df["totals_transactionRevenue"].values
        val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
        #print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
        val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
        print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
        


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn; 
from sklearn import neighbors, datasets

import pylab as pl

from sklearn.decomposition import PCA
pca = PCA(n_components=15)
pca.fit(dev_X)
dev_X_reduced = pca.transform(X)
print("Reduced dataset shape:", dev_X_reduced.shape)

pca.fit(val_X)
val_X_reduced = pca.transform(X)
print("Reduced dataset shape:", val_X_reduced.shape)

pca.fit(test_X)
test_X_reduced = pca.transform(X)
print("Reduced dataset shape:", test_X_reduced.shape)


dev_X = dev_X.iloc[:, ]

cols = ['']


# In[ ]:


import xgboost as xgb

dtrain = xgb.DMatrix(dev_X, label=dev_y)
dtest = xgb.DMatrix(val_X)

#set parameters for xgboost
params = {'max_depth':7,
          'eta':1,
          'silent':1,
          'objective':'reg:linear',
          'eval_metric':'rmse',
          'learning_rate':0.1
         }
num_rounds = 100

xb = xgb.train(params, dtrain, num_rounds)

y_pred_xgb = xb.predict(dtest)


# In[ ]:


rmse = np.sqrt(mean_squared_error(val_y, y_pred_xgb))
print("RMSE: %f" % (rmse))

print("Mean squared error: %.2f"
      % mean_squared_error(val_y, y_pred_xgb))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(val_y, y_pred_xgb))


# In[ ]:


from sklearn import metrics
y_pred_xgb[y_pred_xgb<0] = 0
val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
val_pred_df["transactionRevenue"] = val_df["totals_transactionRevenue"].values
val_pred_df["PredictedRevenue"] = np.expm1(y_pred_xgb)
#print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))


# In[ ]:


dtest_actual = xgb.DMatrix(test_X)
actual_testdata_pred_xgb_params = xb.predict(dtest_actual)

sub_df = pd.DataFrame({"fullVisitorId":test_id})
actual_testdata_pred_xgb_params[actual_testdata_pred_xgb_params<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(actual_testdata_pred_xgb_params)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("new_test1_xgb.csv", index=False)


# In[ ]:


dtest_actual = xgb.DMatrix(test_X)
actual_testdata_pred_xgb_params = xb.predict(dtest_actual)

#EXPORTING PREDICTIONS TO CSV

# key = pd.DataFrame(test[test.columns[0:1]])

# key['fare_amount'] = actual_testdata_pred_xgb_params
# key.to_csv('test_predictions_xgb_params.csv')

