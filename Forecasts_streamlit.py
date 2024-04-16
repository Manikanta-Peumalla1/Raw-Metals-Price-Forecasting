# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:31:00 2024

@author: MK PERUMALLA
"""

import pandas as pd
import pickle
import streamlit as st
from sqlalchemy import create_engine


FN_model=pickle.load(open('Ferro Nickel.pickle','rb'))
Aluminium_model=pickle.load(open('Aluminium.pickle','rb'))
Molybdenum_model=pickle.load(open('Molybdenum.pickle','rb'))
Vanadium_model=pickle.load(open('Vanadium.pickle','rb'))
Graphite_model=pickle.load(open('Graphite.pickle','rb'))
Manganese_model=pickle.load(open('Manganese.pickle','rb'))
Fluorite_model=pickle.load(open('Fluorite.pickle','rb'))

def predict(data, user, pw, db):
    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
    if 'Metal Name' in data.columns:
        data.rename(columns={'Metal Name':'Metal_Name'},inplace=True)
    data['Metal_Name']=data['Metal_Name'].str.strip()
    data.drop(data[data['Metal_Name']=='Magnesium'].index,axis=0,inplace=True)

    if data['Price'].isnull().sum()>0:
        data.set_index('Month',inplace=True)    
        data['Price']=data['Price'].interpolate(method='linear') 
        data.reset_index(inplace=True)
    
    predictions_FN=pd.DataFrame(list(FN_model.predict(start=data[data['Metal_Name']=='Ferro Nickel']['Price'].index[0],end=data[data['Metal_Name']=='Ferro Nickel']['Price'].index[-1])), columns = ['Forecasted Price'])
    
    predictions_Aluminium=pd.DataFrame(list(Aluminium_model.predict(start=data[data['Metal_Name']=='Aluminium']['Price'].index[0],end=data[data['Metal_Name']=='Aluminium']['Price'].index[-1])), columns = ['Forecasted Price'])
    
    predictions_Molybdenum=pd.DataFrame(list(Molybdenum_model.predict(start=data[data['Metal_Name']=='Molybdenum']['Price'].index[0],end=data[data['Metal_Name']=='Molybdenum']['Price'].index[-1])), columns = ['Forecasted Price'])
    
    prediction_Vanadium=pd.DataFrame(list(Vanadium_model.predict(start=data[data['Metal_Name']=='Vanadium']['Price'].index[0],end=data[data['Metal_Name']=='Vanadium']['Price'].index[-1])),columns = ['Forecasted Price'])       
    
    predictions_Graphite=pd.DataFrame(list(Graphite_model.predict(start=data[data['Metal_Name']=='Graphite']['Price'].index[0],end=data[data['Metal_Name']=='Graphite']['Price'].index[-1])), columns = ['Forecasted Price'])
 
    predictions_Manganese=pd.DataFrame(list(Manganese_model.predict(start=data[data['Metal_Name']=='Manganese']['Price'].index[0],end=data[data['Metal_Name']=='Manganese']['Price'].index[-1])), columns = ['Forecasted Price'])
    
    predictions_Fluorite=pd.DataFrame(list(Fluorite_model.predict(start=data[data['Metal_Name']=='Fluorite']['Price'].index[0],end=data[data['Metal_Name']=='Fluorite']['Price'].index[-1])), columns = ['Forecasted Price'])
    
    predictions = pd.concat([predictions_FN,
                             predictions_Aluminium,
                             predictions_Molybdenum,
                             prediction_Vanadium,
                             predictions_Graphite,
                             predictions_Manganese,
                             predictions_Fluorite],
                            axis=0)
    predictions.reset_index(inplace=True,drop=True)
    final = pd.concat([data,predictions], axis = 1,ignore_index=True) 
    final.columns=['Month','Metal_Name','Price','Forecasted_Price']
    final.reset_index(drop=True,inplace=True)
    forecast={'Ferro Nickel':list(FN_model.forecast(steps=6)),
              'Aluminium':list(Aluminium_model.forecast(steps=6)),
              'Molybdenum':list(Molybdenum_model.forecast(steps=6)),
              'Vanadium':list(Vanadium_model.forecast(steps=6)),
              'Graphite':list(Graphite_model.forecast(steps=6)),
              'Manganese':list(Manganese_model.forecast(steps=6)),
              'Fluorite':list(Fluorite_model.forecast(steps=6))
              }
    forecasting=pd.DataFrame(forecast)
    #final.to_sql('Price_forecasts', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

    return final,forecasting


def main():  

    st.title("Raw Metals Price Forecasting")
    st.sidebar.title("Forecasting")
    html_temp = """
    <div style="background-color:lightblue;padding:20px;border-radius:10px;text-align:center;">
    <h2 style="color:navy;"> Raw Metal Price Predictions </h2>
    </div>

    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    
    uploadedFile = st.sidebar.file_uploader("Choose a file", type = ['csv', 'xlsx'], accept_multiple_files = False, key = "fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame(uploadedFile)
                
    else:
        st.sidebar.warning("You need to upload a csv or excel file.")
    
    html_temp = """
    <div style="background-color:lightblue;padding:20px;border-radius:10px;text-align:center;">
    <h2 style="color:navy;"> Raw Metal Price Predictions </h2>
    </div>

    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "root")
    pw = st.sidebar.text_input("password", "mani-123")
    db = st.sidebar.text_input("database", "Raw_materials_Forecasting")
    
    result = ""
    
   
    if st.button("Predict"):
        result = predict(data, user, pw, db)[0]
        import seaborn as sns
        cm = sns.light_palette("yellow", as_cmap = True)
        st.table(result.style.background_gradient(cmap = cm).set_precision(2))                          
    html_temp = """
    <div style="background-color:lightblue;padding:20px;border-radius:10px;text-align:center;">
    <h2 style="color:navy;"> Raw Metal Price Predictions </h2>
    </div>
     
     """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    if st.button("Forecast"):
        result = predict(data, user, pw, db)[1]
        import seaborn as sns
        cm = sns.light_palette("green", as_cmap = True)
        st.table(result.style.background_gradient(cmap = cm).set_precision(2)) 
                           
if __name__=='__main__':
    main()


