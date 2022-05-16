#Step 1: Importing libraries

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
import squarify

#Options

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#Step 2: Reading the dataset

df__rfm  = pd.read_csv("flo_data_20k.csv")
df_rfm= df__rfm.copy()

#Step 3: Data Understanding

#First 10 rows
df_rfm.head

df_rfm.info()

df_rfm.shape # (19945 rows, 12 variables)

df_rfm.isnull().sum() #there is no missing value

df_rfm["interested_in_categories_12"].nunique()

df_rfm["interested_in_categories_12"].value_counts().head() # There are 2135 values in interested_in_categories_12 that are actually empty.

#I wanna eliminate for these rows

df_rfm[df_rfm["interested_in_categories_12"] == "[]"] #2135 rows

df_rfm = df_rfm[df_rfm["interested_in_categories_12"].map(lambda d: len(d)) > 2]

df_rfm.shape # (17810, 12)

df_rfm["order_channel"].nunique()
df_rfm["order_channel"].value_counts()

#Omnichannel, stands for customers who shop both online and offline.

#for omnichannel total over (online + offline)
df_rfm["Omnichannel_total"] = df_rfm["order_num_total_ever_online"] + df_rfm["order_num_total_ever_offline"]

#total cost for omnichannel
df_rfm["Omnichannel_total_cost"] = df_rfm["customer_value_total_ever_offline"] + df_rfm["customer_value_total_ever_online"]

#convert to dtypes
cols=["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]
df_rfm[cols] = df_rfm[cols].apply(pd.to_datetime)

#Total spend by customers, both online and offline
df_rfm.groupby("master_id")["Omnichannel_total_cost"].sum().sort_values(ascending=False).head()

#Total number of purchases, both online and offline
df_rfm.groupby("master_id")["Omnichannel_total"].sum().sort_values(ascending=False).head()

#Total number of purchases for 12 months
total_purchases2 = df_rfm.groupby("interested_in_categories_12")["Omnichannel_total"].sum().sort_values(ascending=False).reset_index().head()

plt.figure(figsize=(10,8))
sns.barplot(data=total_purchases2,x='interested_in_categories_12',y='Omnichannel_total')
plt.show(block=True)

#Total cost of purchases for 12 months
total_cost = df_rfm.groupby("interested_in_categories_12")["Omnichannel_total_cost"].sum().sort_values(ascending=False).reset_index().head()

plt.figure(figsize=(10,8))
sns.barplot(data=total_cost,x='interested_in_categories_12',y='Omnichannel_total_cost')
plt.show(block=True)

#Step 4: RFM Metrics

df_rfm.head()

#Last analysis date
df_rfm["last_order_date"].max() #Timestamp('2021-05-30 00:00:00')

#Recency Date (2 days after the last analysis date)
recency_date = dt.datetime(2021,6,1)
type(recency_date)

#Frequency metric over Omnichannel_total variable ,
#The Monetary metric was also calculated over Omnichannel_total_cost

rfm = df_rfm.groupby('master_id').agg({'last_order_date': lambda last_order_date: (recency_date - last_order_date.max()).days,
                                     'Omnichannel_total': lambda Omnichannel_total: Omnichannel_total.sum(),
                                     'Omnichannel_total_cost': lambda Omnichannel_total_cost: Omnichannel_total_cost.sum()})


rfm.columns = ['Recency', 'Frequency', 'Monetary']

rfm.describe().T

#Step 5: RFM Scores

#After this step, we need to return the RFM scores to the 1–5 scale. At this point, the qcut() function can be used

rfm["recency_score"]=pd.qcut(rfm['Recency'],5,labels=[5,4,3,2,1])

rfm["frequency_score"] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm.head()

#Concat Recency,Frequency,Monetary Metrics

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(str))

rfm[rfm["RFM_SCORE"]=="555"].head()

#Final Step: Creating Segments

#We're going to use a regular expression here. The seg_map we created as a dictionary.

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

#Segmap include dataframe

rfm['Segment'] = rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str)

rfm['Segment'] = rfm['Segment'].replace(seg_map, regex=True)

rfm.head()

rfm[["Segment", "Recency", "Frequency", "Monetary"]].groupby("Segment").agg(["mean", "count"])

#Segmentation Graph
segments = rfm['Segment'].value_counts().sort_values(ascending=False)

fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(12, 8)
squarify.plot(sizes=segments,
              label=['hibernating',
                     'at_Risk',
                     'cant_loose',
                     'about_to_sleep',
                     'need_attention',
                     'loyal_customers',
                     'promising',
                     'new_customers',
                     'potential_loyalists',
                     'champions'], alpha=.7,color=["purple","orange","blue", "green","grey"],pad=True)
plt.title("RFM Segmentation",fontsize=12,fontweight="light")
plt.axis('off')
plt.show(block=True)

"""
CASE 1:
A new women's shoe brand will be included. 
The target audience (champions, loyal_customers) and women are determined as shoppers. 
We need access to the id numbers of these customers.
"""

rfm[rfm["Segment"] == "loyal_customers"] #2980

rfm[rfm["Segment"] == "champions"] #1900

segment_a = a= rfm[(rfm["Segment"]=="champions") | (rfm["Segment"]=="loyal_customers")]

segment_b = df_rfm[(df_rfm["interested_in_categories_12"]).str.contains("KADIN")] #7603

merge_case1= pd.merge(segment_a,segment_b[["interested_in_categories_12","master_id"]],on=["master_id"])

merge_case1.columns

merge_case1= merge_case1.drop(merge_case1.loc[:,'Recency':'interested_in_categories_12'].columns,axis=1)

#Turn the csv format
merge_case1.to_csv("customer_information_1.csv",index=False)

"""
CASE 2:
A 40% discount on men's and children's products is planned. 
The target audience is (cant_loose, about_to_sleep, new_customers). 
We need to access the id numbers of these customers.
"""

segment_c = rfm[(rfm["Segment"]=="cant_loose") | (rfm["Segment"]=="about_to_sleep") | (rfm["Segment"]=="new_customers")]

segment_d = df_rfm[(df_rfm["interested_in_categories_12"]).str.contains("ERKEK|COCUK")]

merge_case2= pd.merge(segment_c,segment_d[["interested_in_categories_12","master_id"]],on=["master_id"])

merge_case2= merge_case2.drop(merge_case2.loc[:,'Recency':'interested_in_categories_12'].columns,axis=1)

#Turn the csv format

merge_case2.to_csv("customer_information_2.csv",index=False)

segments = rfm['Segment'].value_counts().sort_values(ascending=False)




















