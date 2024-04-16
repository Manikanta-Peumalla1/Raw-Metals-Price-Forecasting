# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 15:35:50 2024

@author: MK PERUMALLA
"""





# data handling libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import warnings

# modeling libraries
from statsmodels.tsa.holtwinters import SimpleExpSmoothing #simple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # for holt's winter model
from statsmodels.tsa.holtwinters import Holt # for holts model
from statsmodels.tsa.api import AutoReg #for ar
from statsmodels.tsa.arima.model import ARIMA # for ar,ma,arma,arima
import pmdarima as pm # for auto arima
from sklearn.metrics import mean_absolute_error
import pickle




warnings.filterwarnings('ignore')


user='root' # username of database
pw='mani-123' #password
db='Raw_materials_Forecasting' #database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
sql = '''
         select * from raw_metals_data_processed;
  '''

df = pd.read_sql_query(sql, con = engine)

# stripping whitespaces in Metal_Name
df['Metal_Name']=df['Metal_Name'].str.strip()

# dropping Magnesium records
df.drop(df[df['Metal_Name']=='Magnesium'].index,axis=0,inplace=True)
   
# Let's create Function to calculate Mean Absalute Persentage error
def MAPE(pred, actual):
    temp = np.abs((pred - actual)/actual)*100
    return np.mean(temp)


# Group your data by 'Metal_Name'
grouped_data = df.groupby('Metal_Name')

# get data for Fluorite
grouped_data.get_group('Fluorite')

metals=df.Metal_Name.unique()
#------------------------------Modeling----------------------------------------


for i,j in enumerate(df['Metal_Name'].unique()):
    print(i,j)
    plt.subplot(4,2,i+1)
    plt.title(j)
    df[df['Metal_Name']==j]['Price'].plot()


"""
#----------------------- Moving Averages Model ---------------------------


preds=df['Price'].head(int(df.shape[0]*0.8)).rolling(4).mean()
test=df['Price'].tail(df.shape[0]-int(df.shape[0]*0.8))
MAPE(preds, test)
MA={}
for name,group in grouped_data:
    print("Building Moving Average's model for:", name)
    preds=group['Price'].head(int(group.shape[0]*0.8)).rolling(4).mean()
    MA[name]=MAPE(preds.tail(group.shape[0]-int(group.shape[0]*0.8)),group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)))
    
MA

preds=group['Price'].head(int(group.shape[0]*0.8)).rolling(4).mean()

preds.tail(group.shape[0]-int(group.shape[0]*0.8)).shape

group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).shape

MAPE(preds.tail(group.shape[0]-int(group.shape[0]*0.8)).shape, group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).shape)
h=int(group.shape[0]*0.8)
t=group.shape[0]-int(group.shape[0]*0.8)

type()

type(group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)))


"""



#---------------Simple Exponential Smoothing-------------------------


ses={}
for name,group in grouped_data:
    print("Building Simple Exponential model for:", name)
    ses_model = SimpleExpSmoothing(group['Price'].head(int(group.shape[0]*0.8))).fit()
    pred_ses = ses_model.predict(start = group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[0], end = group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[-1])
    ses[name] = MAPE(pred_ses, group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)))

ses


pickle.dump(ses_model,open('ses_model.pickel','wb'))

##---------------------------------Holt's Model----------------------------

holt={}
for name,group in grouped_data:
    print("Building Holt's model for:", name)
    holt_model = Holt(group['Price'].head(int(group.shape[0]*0.8))).fit()
    pred_holt = holt_model.predict(start = group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[0], end = group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[-1])
    holt[name] = MAPE(pred_holt, group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)))

holt



#--------------------------- Holt's Winter ------------------------------



# Holts winter exponential smoothing with additive seasonality and additive trend

hwe_ad={}
for name,group in grouped_data:
    print("Building Holt's winter model for:", name)
    hwe_model_ad=ExponentialSmoothing(group['Price'].head(int(group.shape[0]*0.8)),seasonal='additive',trend='additive',seasonal_periods=6).fit()
    pred_hwe=hwe_model_ad.predict(start=group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[0], end = group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[-1])
    hwe_ad[name]=MAPE(pred_hwe, group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)))

hwe_ad


# Holts winter exponential smoothing with multiplicative seasonality and multiplicative trend

hwe_mul={}
for name,group in grouped_data:
    print("Building Holt's winter model for:", name)
    hwe_model_mul=ExponentialSmoothing(group['Price'].head(int(group.shape[0]*0.8)),seasonal='mul',trend='mul',seasonal_periods=6).fit()
    pred_hwe_mul=hwe_model_mul.predict(start=group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[0], end = group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[-1])
    hwe_mul[name]=MAPE(pred_hwe_mul, group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)))

hwe_mul


hwe_mul_df = pd.DataFrame.from_dict(hwe_mul, orient='index', columns=['MAPE_hwe_mul'])
hwe_add_df = pd.DataFrame.from_dict(hwe_ad, orient='index', columns=['MAPE_hwe_ad'])
holt_df = pd.DataFrame.from_dict(holt, orient='index', columns=['MAPE_holt'])
ses_df=pd.DataFrame.from_dict(ses, orient='index', columns=['MAPE_ses'])

smoothing_models=pd.concat([ses_df,holt_df,hwe_add_df,hwe_mul_df],axis=1)



#-------------------------------- AR MODEL -------------------------------

# AR(6)

ar={}
for name,group in grouped_data:
    print("Building  AR model for:", name)
    ar_model=AutoReg(group['Price'].head(int(group.shape[0]*0.8)), lags=6).fit()
    pred_ar=ar_model.predict(start=group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[0], end = group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[-1],dynamic=False)
    ar[name]=MAPE(pred_ar, group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)))

ar

#-------------------------------- MA MODEL -------------------------------

# MA(4)

ma={}
for name,group in grouped_data:
    print("Building MA model for:", name)
    ma_model=ARIMA(group['Price'].head(int(group.shape[0]*0.8)),order=(0,0,4)).fit()
    pred_ma=ma_model.predict(start=group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[0], end = group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[-1],dynamic=False)
    ma[name]=MAPE(pred_ma, group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)))

ma


#-------------------------------- ARMA MODEL -------------------------------

# ARMA(4,4)

arma={}
for name,group in grouped_data:
    print("Building ARMA model for:", name)
    arma_model=ARIMA(group['Price'].head(int(group.shape[0]*0.8)),order=(4,0,4)).fit()
    pred_arma=arma_model.predict(start=group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[0], end = group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[-1],dynamic=False)
    arma[name]=MAPE(pred_arma, group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)))

arma


#-------------------------------- ARIMA MODEL -------------------------------

# ARIMA(4,2,4)

arima={}
for name,group in grouped_data:
    print("Building ARIMA model for:", name)
    arima_model=ARIMA(group['Price'].head(int(group.shape[0]*0.8)),order=(4,1,4)).fit()
    pred_arima=arima_model.predict(start=group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[0], end = group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[-1],dynamic=False)
    arima[name]=MAPE(pred_arima, group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)))

arima




ar_model_df = pd.DataFrame.from_dict(ar, orient='index', columns=['MAPE_ar'])
ma_model_df = pd.DataFrame.from_dict(ma, orient='index', columns=['MAPE_ma'])
arma_model_df = pd.DataFrame.from_dict(arma, orient='index', columns=['MAPE_arma'])
arima_model_df=pd.DataFrame.from_dict(arima, orient='index', columns=['MAPE_arima'])

classical_models=pd.concat([ar_model_df,ma_model_df,arma_model_df,arima_model_df],axis=1)




all_models=pd.concat([smoothing_models,classical_models],axis=1)






#--------------------------- AUTO ARIMA MODEL -------------------------------


best_models={}

# fitting auto arima to choose best model
for name,group in grouped_data:
    print("Building Auto ARIMA for:", name)
    ar_model = pm.auto_arima(group['Price'].head(int(group.shape[0]*0.8)), start_p = 0, start_q = 0,
                      max_p = 6, max_q = 6, # maximum p and q
                      m = 6,              # frequency of series
                      d = None,           # let model determine 'd'
                      seasonal = True,   # Seasonality
                      start_P = 0,max_P=6, trace = True,
                      error_action = 'warn', stepwise = True)
    best_models[name]=ar_model

ar_model
best_models



#----------------- Best Models From Auto ARIMA ----------------------------

best_model={}
for name,group in grouped_data:
    print("Building ARIMA model for:", name)
    # fitting model
    model=ARIMA(group['Price'].head(int(group.shape[0]*0.8)),order=best_models[name].order,seasonal_order=best_models[name].seasonal_order).fit()
    # forecast on unseen data 
    pred=model.predict(start=group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[0], end = group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)).index[-1],dynamic=False)
    # mean absalute error
    mae=mean_absolute_error(pred, group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8)))
    #throwing best model scores into dictionary named best_model
    best_model[name]=['ARIMA(order={a},seasonal_order={b})'.format(a=best_models[name].order,b=best_models[name].seasonal_order),MAPE(pred, group['Price'].tail(group.shape[0]-int(group.shape[0]*0.8))),mae]
    # saving models for all metals
    pickle.dump(model,open('{a}.pickle'.format(a=name),'wb'))
best_model

forecasts={}
for name,group in grouped_data:
    model=ARIMA(group['Price'].head(int(group.shape[0]*0.8)),order=best_models[name].order,seasonal_order=best_models[name].seasonal_order).fit()
    fore=list(model.forecast(steps=6))
    forecasts[name]=fore
forecasts_df=pd.DataFrame(forecasts)

final_results=pd.DataFrame(best_model)

final_results['index']=['Model','MAPE','MAE']
final_results.set_index('index',inplace=True)


# -----------------------------------The End-------------------------------
grouped_data.groups.keys()



