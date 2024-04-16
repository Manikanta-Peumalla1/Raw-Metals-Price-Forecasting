# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 05:54:16 2024

@author: MK PERUMALLA
"""

# importing all required Packages
import pandas as pd
import numpy as np
import sweetviz
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import missingno as msno 
# Loading Data
data=pd.read_excel("C:/Users/MK PERUMALLA/OneDrive/Desktop/360DigitMG/Project-1/Data/Raw Material(Minerals  Metals) (2).xlsx")

# sql connection establishment
user='root' # username of database
pw='mani-123' #password
db='Raw_materials_Forecasting' #database name

#  creating engine to connect database
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# pushing data into database 
# name should be in lower case

data.to_sql('Raw_Metals_data', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# loading data from database
sql = 'select * from Raw_Metals_data'

df = pd.read_sql_query(sql, con = engine)

print(df)

df.shape #shape of df

df.columns # all columns

dir(df)




# Create an empty DataFrame to store the stacked tails
stacked_data = pd.DataFrame(columns=df.columns)

# Iterate over unique values in the 'Metal_Name' column
for metal_name in df['Metal_Name'].unique():
    # Extract the last 7 rows for each unique metal name and append to stacked_data
    stacked_data = pd.concat([stacked_data, df[df['Metal_Name'] == metal_name].tail(6)])

# Display the resulting stacked DataFrame
print(stacked_data)

stacked_data.to_csv('test_data.csv')

"""   

-------------------------Business Insights-------------------------------

"""



############################ Before Pre-Processing ##########################


data.Metal_Name.value_counts()

plt.subplot(1,2,1)
# Pie chart
plt.title('Metals Pie Chart')
plt.pie(data.Metal_Name.value_counts(sort=True),labels=data.Metal_Name.value_counts(sort=True).index,autopct='%1.1f%%',data=df)

plt.subplot(1,2,2)
# count plot
plt.title('Metals count plot')
sns.countplot(data=data,x='Metal_Name')


"""

-------------------------Statistical Insights-----------------------------

"""

############################ Before Pre-Processing ##########################


# Checking Distribution 


sns.displot(data['Price']) # on whole data

# distribution specified Metal Only 

for i,j in enumerate(data.Metal_Name.unique()):
    print(i,j)
    plt.subplot(4,2,i+1)
    plt.title(j+"'s Distribution")
    sns.distplot(data[data['Metal_Name']==j].Price)
    plt.xlabel(i,size=15)
    plt.tight_layout()

# Checking Missing Values

data.isnull().sum()

print("Missing values in each Metal\n")
for i in data.Metal_Name.unique():
    print(i," : " ,data[data['Metal_Name']==i]['Price'].isnull().sum())

# visualizing Missing Values

for i,j in enumerate(data.Metal_Name.unique()):
    print(i,j)
    plt.subplot(4,2,i+1)
    plt.title(" Missing Records in "+j)
    msno.matrix(data)
    plt.xlabel(i,size=15)
    plt.tight_layout()


# checking for outliers
sns.boxplot(x="Price",data=df) # on Whole data

# Checking Outliers in each Metal
for i,j in enumerate(data.Metal_Name.unique()):
    print(i,j)
    plt.subplot(4,2,i+1)
    plt.title(j+"'s Boxplot")
    sns.boxplot(x="Price",data=data[data['Metal_Name']==j])
    plt.xlabel(i,size=15)
    plt.tight_layout()



# Statistical Quantiles

# On  whole data    
whole_insights_1=data.describe()

# On each Metal
insights_1=data.groupby('Metal_Name').describe()

# variance
int(data['Price'].var())

# On each Metal
for i in data.Metal_Name.unique():
    print(" Variance of "+i," : " ,data[data['Metal_Name']==i]['Price'].var())


# Visualizing Auto correlation
for i,j in enumerate(data.Metal_Name.unique()):
    print(i,j)
    plt.subplot(4,2,i+1)
    plt.title(j+"'s Distribution")
    plt.acorr(data[data['Metal_Name']==j].Price,maxlags=10)
    plt.xlabel(i,size=15)
    plt.tight_layout()

# Auto correlation on whole data at lag 1,2,3 

lag1_autocorrelation_1 = data['Price'].autocorr(lag=1)  # For lag 1
print("Autocorrelation at lag 1:", lag1_autocorrelation_1)

lag2_autocorrelation_1 = data['Price'].autocorr(lag=2)  # For lag 2
print("Autocorrelation at lag 2:", lag2_autocorrelation_1)

lag3_autocorrelation_1 = data['Price'].autocorr(lag=3)  # For lag 2
print("Autocorrelation at lag 2:", lag3_autocorrelation_1)

# Auto correlation on each metal data at lag 1

for i in data.Metal_Name.unique():
    print(" Auto correlation of " +i," at lag 1 is : " ,data[data['Metal_Name']==i]['Price'].autocorr(lag=1))
    print(" Auto correlation of " +i," at lag 2 is : " ,data[data['Metal_Name']==i]['Price'].autocorr(lag=2))
    print(" Auto correlation of " +i," at lag 3 is : " ,data[data['Metal_Name']==i]['Price'].autocorr(lag=3))


############################## Data Pre Processsing ########################


df.loc[283,'Price']=np.nan

df.set_index('Month',inplace=True)

df['Price']=df['Price'].interpolate(method='linear')

df.isnull().sum()

df.reset_index(inplace=True)

df.sort_values('Month')

df.to_sql('Raw_Metals_data_processed', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

df.to_csv('Raw_Metals_data_processed.csv', index=False)


"""

------------------------- Business Insights ----------------------------------

"""

############################ After Pre-Processing ##########################

df.Metal_Name.value_counts()

plt.subplot(1,2,1)
# Pie chart
plt.title('Metals Pie Chart')
plt.pie(df.Metal_Name.value_counts(sort=True),labels=df.Metal_Name.value_counts(sort=True).index,autopct='%1.1f%%',data=df)

plt.subplot(1,2,2)
# count plot
plt.title('Metals count plot')
sns.countplot(data=df,x='Metal_Name')


plt.plot(data.Month,data.Price)


"""

------------------------- Statistical Insights ----------------------------------

"""

############################ After Pre-Processing ##########################


# Checking Distribution 


sns.displot(df['Price']) # on whole data

# distribution specified Metal Only 

for i,j in enumerate(df.Metal_Name.unique()):
    print(i,j)
    plt.subplot(4,2,i+1)
    plt.title(j+"'s Distribution")
    sns.distplot(df[df['Metal_Name']==j].Price)
    plt.xlabel(i,size=15)
    plt.tight_layout()

# Checking Missing Values

df.isnull().sum()

print("Missing values in each Metal\n")
for i in df.Metal_Name.unique():
    print(i," : " ,df[df['Metal_Name']==i]['Price'].isnull().sum())

# visualizing Missing Values

for i,j in enumerate(df.Metal_Name.unique()):
    print(i,j)
    plt.subplot(4,2,i+1)
    plt.title(" Missing Records in "+j)
    msno.matrix(df)
    plt.xlabel(i,size=15)
    plt.tight_layout()


# checking for outliers
sns.boxplot(x="Price",data=df) # on Whole data

# Checking Outliers in each Metal
for i,j in enumerate(df.Metal_Name.unique()):
    print(i,j)
    plt.subplot(4,2,i+1)
    plt.title(j+"'s Boxplot")
    sns.boxplot(x="Price",data=df[df['Metal_Name']==j])
    plt.xlabel(i,size=15)
    plt.tight_layout()

# Visualizing Auto correlation
for i,j in enumerate(df.Metal_Name.unique()):
    print(i,j)
    plt.subplot(4,2,i+1)
    plt.title(j+"'s Distribution")
    plt.acorr(df[df['Metal_Name']==j].Price,maxlags=10)
    plt.xlabel(i,size=15)
    plt.tight_layout()




# Statistical Quantiles

# On  whole data
whole_insights_2=df.describe()

# On each Metal Data
insights_2=df.groupby('Metal_Name').describe()



lag1_autocorrelation_2 = df['Price'].autocorr(lag=1)  # For lag 1
print("Autocorrelation at lag 1:", lag1_autocorrelation_2)

lag2_autocorrelation_2 = df['Price'].autocorr(lag=2)  # For lag 2
print("Autocorrelation at lag 2:", lag2_autocorrelation_2)

lag3_autocorrelation_2 = df['Price'].autocorr(lag=3)  # For lag 2
print("Autocorrelation at lag 2:", lag3_autocorrelation_2)

for i in df.Metal_Name.unique():
    print(" Auto correlation of " +i," at lag 1 is : " ,df[df['Metal_Name']==i]['Price'].autocorr(lag=1))
    print(" Auto correlation of " +i," at lag 2 is : " ,df[df['Metal_Name']==i]['Price'].autocorr(lag=2))
    print(" Auto correlation of " +i," at lag 3 is : " ,df[df['Metal_Name']==i]['Price'].autocorr(lag=3))






