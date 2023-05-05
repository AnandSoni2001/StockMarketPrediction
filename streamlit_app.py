#Import Libraries
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from yahoo_fin import stock_info
from yahoo_fin.stock_info import *
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
import joblib
import yfinance as yf
import time
import requests
from bs4 import BeautifulSoup

#Heading
st.title('Research Project on Stock Market Analysis and Prediction')
st.write("#")

#TCS Data Taken
tcsdaily = stock_info.get_data("TCS.NS", interval="1d")
tcsmonthly= stock_info.get_data("TCS.NS", interval="1mo")
tcsyearly = pd.read_csv('data/tcs-yearly.csv')

#Reliance Data Taken
reldaily = stock_info.get_data("RELIANCE.NS", interval="1d")
relmonthly= stock_info.get_data("RELIANCE.NS", interval="1mo")
relyearly = pd.read_csv('data/relianceind-yearly.csv')

#Infosys Data Taken
infdaily = stock_info.get_data("INFY.NS", interval="1d")
infmonthly= stock_info.get_data("INFY.NS", interval="1mo")
infyearly = pd.read_csv('data/infosys-yearly.csv')

#Select Box
comp = st.selectbox('Select a Company from the below options :', ('Tata Consultancy Services - TCS', 'Reliance Industries - RELIANCE', 'Infosys - INFY'))

if comp == 'Tata Consultancy Services - TCS':
    
    page = requests.get('https://groww.in/stocks/tata-consultancy-services-ltd')
    soup=BeautifulSoup(page.content,'html.parser')
    fund=soup.find_all('td',class_="ft785Value")

    #Fundamental Values
    pb = float(fund[4].text)
    pe = float(fund[2].text)
    de = float(fund[8].text)
    div = float(fund[5].text.replace('%',''))
    roe = float(fund[1].text.replace('%',''))
    indpe = float(fund[6].text)
    
    col1, col2, col3, col4 = st.columns(4)
    x = round(stock_info.get_live_price("TCS.NS"),2)
    y = round(tcsdaily['close'].iloc[-2],2)
    tcs = get_stats('TCS.NS')['Value']
    
    col1.metric(label="Market Price", value=x, delta = round(x-y,2))
    col2.metric(label="52 Week High", value=tcs[3])
    col3.metric(label="52 Week Low", value=tcs[4])
    col4.metric(label="Return on Equity", value=tcs[34])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label='P/B Ratio', value=pb)
    col2.metric(label="P/E Ratio", value=pe)
    col3.metric(label='Industry P/E', value=indpe)
    col4.metric(label="Debt to Equity", value=de)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label='Previous Close', value=y)
    col2.metric(label="Book Value Per Share", value=tcs[48])
    col3.metric(label='Earning Per Share', value=tcs[41])
    col4.metric(label="Dividend Yield", value=tcs[22])


if comp == 'Reliance Industries - RELIANCE':
    
    page = requests.get('https://groww.in/stocks/reliance-industries-ltd')
    soup=BeautifulSoup(page.content,'html.parser')
    fund=soup.find_all('td',class_="ft785Value")

    #Fundamental Values
    pb = float(fund[4].text)
    pe = float(fund[2].text)
    de = float(fund[8].text)
    div = float(fund[5].text.replace('%',''))
    roe = float(fund[1].text.replace('%',''))
    indpe = float(fund[6].text)
    
    col1, col2, col3, col4 = st.columns(4)
    x = round(stock_info.get_live_price("RELIANCE.NS"),2)
    y = round(reldaily['close'].iloc[-2],2)
    rel = get_stats('RELIANCE.NS')['Value']
    col1.metric(label="Market Price", value=x, delta = round(x-y,2))
    col2.metric(label="52 Week High", value=rel[3])
    col3.metric(label="52 Week Low", value=rel[4])
    col4.metric(label="Return on Equity", value='8.21%')
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label='P/B Ratio', value=pb)
    col2.metric(label="P/E Ratio", value=pe)
    col3.metric(label='Industry P/E', value=indpe)
    col4.metric(label="Debt to Equity", value=de)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label='Previous Close', value=y)
    col2.metric(label="Book Value Per Share", value=float(fund[7].text))
    col3.metric(label='Earning Per Share', value=float(fund[3].text))
    col4.metric(label="Dividend Yield", value=div)

if comp == 'Infosys - INFY':
    
    page = requests.get('https://groww.in/stocks/infosys-ltd')
    soup=BeautifulSoup(page.content,'html.parser')
    fund=soup.find_all('td',class_="ft785Value")

    #Fundamental Values
    pb = float(fund[4].text)
    pe = float(fund[2].text)
    de = float(fund[8].text)
    div = float(fund[5].text.replace('%',''))
    roe = float(fund[1].text.replace('%',''))
    indpe = float(fund[6].text)
    
    col1, col2, col3, col4 = st.columns(4)
    x = round(stock_info.get_live_price("INFY.NS"),2)
    y = round(infdaily['close'].iloc[-2],2)
    inf = get_stats('INFY.NS')['Value']
    col1.metric(label="Market Price", value=x, delta = round(x-y,2))
    col2.metric(label="52 Week High", value=inf[3])
    col3.metric(label="52 Week Low", value=inf[4])
    col4.metric(label="Return on Equity", value=inf[34])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label='P/B Ratio', value=pb)
    col2.metric(label="P/E Ratio", value=pe)
    col3.metric(label='Industry P/E', value=indpe)
    col4.metric(label="Debt to Equity", value=de)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label='Previous Close', value=y)
    col2.metric(label="Book Value Per Share", value=inf[48])
    col3.metric(label='Earning Per Share', value=inf[41])
    col4.metric(label="Dividend Yield", value=inf[22])

#Tab for Hist Data
st.write("#")
st.subheader('Historic data : ')
option1, option2, option3 = st.tabs(["Daily", "Monthly", "Yearly"])

cl1, cl2, cl3, cl4 = st.columns(4)
with cl1:
    ag1 = st.checkbox('Close', value='True')
with cl2:
    ag2 = st.checkbox('Open', value='True')
with cl3:
    ag3 = st.checkbox('High', value='True')
with cl4:
    ag4 = st.checkbox('Low', value='True')
            
with option1:
    opt = st.radio("Select timelength :", ('All Time', '1 Week', '1 Month', '1 Year'))
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    
    if comp == 'Tata Consultancy Services - TCS':
        if opt=='All Time' :
            fig = px.line(tcsdaily, y='close',markers=False, title='Tata Consultancy Services daily data of all time')  
        if opt=='1 Week' :
            fig = px.line(tcsdaily.tail(5), y='close',markers=False, title='Tata Consultancy Services daily data of 1 week')   
        if opt=='1 Month' :
            fig = px.line(tcsdaily.tail(20), y='close',markers=False, title='Tata Consultancy Services daily data of 1 month')     
        if opt=='1 Year' :
            fig = px.line(tcsdaily.tail(251), y='close',markers=False, title='Tata Consultancy Services daily data of 1 year') 
        st.plotly_chart(fig, use_container_width=True)
        
        fig = go.Figure()
        if(ag1):
            fig.add_trace(go.Scatter(x=tcsdaily.index,y=tcsdaily['close'], name='Closing'))
        if(ag2):
            fig.add_trace(go.Scatter(x=tcsdaily.index,y=tcsdaily['open'], name = 'Opening', line=dict(color='yellow')))
        if(ag3):
            fig.add_trace(go.Scatter(x=tcsdaily.index,y=tcsdaily['high'], name = 'High', line=dict(color='green')))
        if(ag4):
            fig.add_trace(go.Scatter(x=tcsdaily.index,y=tcsdaily['low'], name = 'Low', line=dict(color='red')))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price', title='Comparing other relevant parameters along close')
        st.plotly_chart(fig, use_container_width=True, title='Comparing other relevant parameters')

    if comp == 'Infosys - INFY':
        if opt=='All Time' :
            fig = px.line(infdaily, y='close',markers=False, title='Infosys daily data of all time')  
        if opt=='1 Week' :
            fig = px.line(infdaily.tail(5), y='close',markers=False, title='Infosys daily data of 1 week')   
        if opt=='1 Month' :
            fig = px.line(infdaily.tail(20), y='close',markers=False, title='Infosys daily data of 1 month')     
        if opt=='1 Year' :
            fig = px.line(infdaily.tail(251), y='close',markers=False, title='Infosys daily data of 1 year') 
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()   
        if(ag1):
            fig.add_trace(go.Scatter(x=infdaily.index, y=infdaily['close'], name='Closing', line=dict(color='blue')))
        if(ag2):
            fig.add_trace(go.Scatter(x=infdaily.index,y=infdaily['open'], name = 'Opening', line=dict(color='yellow')))
        if(ag3):
            fig.add_trace(go.Scatter(x=infdaily.index,y=infdaily['high'], name = 'High', line=dict(color='green')))
        if(ag4):
            fig.add_trace(go.Scatter(x=infdaily.index,y=infdaily['low'], name = 'Low', line=dict(color='red')))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price', title='Comparing other relevant parameters')
        st.plotly_chart(fig, use_container_width=True)

    if comp == 'Reliance Industries - RELIANCE':
        if opt=='All Time' :
            fig = px.line(reldaily, y='close',markers=False, title='Reliance Industries daily data of all time')  
        if opt=='1 Week' :
            fig = px.line(reldaily.tail(5), y='close',markers=False, title='Reliance Industries daily data of 1 week')   
        if opt=='1 Month' :
            fig = px.line(reldaily.tail(20), y='close',markers=False, title='Reliance Industries daily data of 1 month')     
        if opt=='1 Year' :
            fig = px.line(reldaily.tail(251), y='close',markers=False, title='Reliance Industries daily data of 1 year') 
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        if(ag1):
            fig.add_trace(go.Scatter(x=reldaily.index, y=reldaily['close'], name='Closing', line=dict(color='blue')))
        if(ag2):
            fig.add_trace(go.Scatter(x=reldaily.index,y=reldaily['open'], name = 'Opening', line=dict(color='yellow')))
        if(ag3):
            fig.add_trace(go.Scatter(x=reldaily.index,y=reldaily['high'], name = 'High', line=dict(color='green')))
        if(ag4):
            fig.add_trace(go.Scatter(x=reldaily.index,y=reldaily['low'], name = 'Low', line=dict(color='red')))
        fig.update_layout(xaxis_title='Date', yaxis_title='Price', title='Comparing other relevant parameters along close')
        st.plotly_chart(fig, use_container_width=True)

with option2:
    if comp == 'Tata Consultancy Services - TCS':
        fig = px.line(tcsmonthly,y='close', markers=False, title='Tata Consultancy Services monthly data')
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        if(ag1):
            fig.add_trace(go.Scatter(x=tcsmonthly.index,y=tcsmonthly['close'], name='Closing', line=dict(color='blue')))
        if(ag2):
            fig.add_trace(go.Scatter(x=tcsmonthly.index,y=tcsmonthly['open'], name = 'Opening', line=dict(color='yellow')))
        if(ag3):
            fig.add_trace(go.Scatter(x=tcsmonthly.index,y=tcsmonthly['high'], name = 'High', line=dict(color='green')))
        if(ag4):
            fig.add_trace(go.Scatter(x=tcsmonthly.index,y=tcsmonthly['low'], name = 'Low', line=dict(color='red')))
        fig.update_layout(xaxis_title='Month', yaxis_title='Price', title='Comparing other relevant parameters')
        st.plotly_chart(fig, use_container_width=True)

    if comp == 'Infosys - INFY':
        fig = px.line(infmonthly, y='close',markers=False, title='Infosys monthly data')
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        if(ag1):
            fig.add_trace(go.Scatter(x=infmonthly.index, y=infmonthly['close'], name='Closing', line=dict(color='blue')))
        if(ag2):
            fig.add_trace(go.Scatter(x=infmonthly.index,y=infmonthly['open'], name = 'Opening', line=dict(color='yellow')))
        if(ag3):
            fig.add_trace(go.Scatter(x=infmonthly.index,y=infmonthly['high'], name = 'High', line=dict(color='green')))
        if(ag4):
            fig.add_trace(go.Scatter(y=infmonthly['low'], name = 'Low', line=dict(color='red')))
        fig.update_layout(xaxis_title='Month', yaxis_title='Price', title='Comparing other relevant parameters')
        st.plotly_chart(fig, use_container_width=True)

    if comp == 'Reliance Industries - RELIANCE':
        fig = px.line(relmonthly, y='close',markers=False, title='Reliance Industries monthly data')
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        if(ag1):
            fig.add_trace(go.Scatter(x=relmonthly.index,y=relmonthly['close'], name='Closing', line=dict(color='blue')))
        if(ag2):
            fig.add_trace(go.Scatter(x=relmonthly.index,y=relmonthly['open'], name = 'Opening', line=dict(color='yellow')))
        if(ag3):
            fig.add_trace(go.Scatter(x=relmonthly.index,y=relmonthly['high'], name = 'High', line=dict(color='green')))
        if(ag4):
            fig.add_trace(go.Scatter(x=relmonthly.index,y=relmonthly['low'], name = 'Low', line=dict(color='red')))
        fig.update_layout(xaxis_title='Month', yaxis_title='Price', title='Comparing other relevant parameters')
        st.plotly_chart(fig, use_container_width=True)

with option3:
    if comp == 'Tata Consultancy Services - TCS':
        fig = px.line(tcsyearly, x='Year', y='Close Price',markers=True, title='Tata Consultancy Services Yearly Data from 2004')
        st.plotly_chart(fig, use_container_width=True)
        
        fig = go.Figure()
        if(ag1):
            fig.add_trace(go.Scatter(x=tcsyearly['Year'], y=tcsyearly['Close Price'], name='Closing', line=dict(color='blue')))
        if(ag2):
            fig.add_trace(go.Scatter(x=tcsyearly['Year'], y=tcsyearly['Open Price'], name = 'Opening', line=dict(color='yellow')))
        if(ag3):
            fig.add_trace(go.Scatter(x=tcsyearly['Year'], y=tcsyearly['High Price'], name = 'High', line=dict(color='green')))
        if(ag4):
            fig.add_trace(go.Scatter(x=tcsyearly['Year'], y=tcsyearly['Low Price'], name = 'Low', line=dict(color='red')))
        fig.update_layout(xaxis_title='Year', yaxis_title='Price', title='Comparing other relevant parameters along close price')
        st.plotly_chart(fig, use_container_width=True, title='Comparing other relevant parameters')

    if comp == 'Infosys - INFY':
        fig = px.line(infyearly, x='Year', y='Close Price',markers=True, title='Infosys Yearly Data from 2004')
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        if(ag1):
            fig.add_trace(go.Scatter(x=infyearly['Year'], y=infyearly['Close Price'], name='Closing', line=dict(color='blue')))
        if(ag2):
            fig.add_trace(go.Scatter(x=infyearly['Year'], y=infyearly['Open Price'], name = 'Opening', line=dict(color='yellow')))
        if(ag3):
            fig.add_trace(go.Scatter(x=infyearly['Year'], y=infyearly['High Price'], name = 'High', line=dict(color='green')))
        if(ag4):
            fig.add_trace(go.Scatter(x=infyearly['Year'], y=infyearly['Low Price'], name = 'Low', line=dict(color='red')))
        fig.update_layout(xaxis_title='Year', yaxis_title='Price', title='Comparing other relevant parameters')
        st.plotly_chart(fig, use_container_width=True)

    if comp == 'Reliance Industries - RELIANCE':
        fig = px.line(relyearly, x='Year', y='Close Price',markers=True, title='Reliance Industries Yearly Data from 2004')
        st.plotly_chart(fig, use_container_width=True)

        fig = go.Figure()
        if(ag1):
            fig.add_trace(go.Scatter(x=relyearly['Year'], y=relyearly['Close Price'], name='Closing', line=dict(color='blue')))
        if(ag2):
            fig.add_trace(go.Scatter(x=relyearly['Year'], y=relyearly['Open Price'], name = 'Opening', line=dict(color='yellow')))
        if(ag3):
            fig.add_trace(go.Scatter(x=relyearly['Year'], y=relyearly['High Price'], name = 'High', line=dict(color='green')))
        if(ag4):
            fig.add_trace(go.Scatter(x=relyearly['Year'], y=relyearly['Low Price'], name = 'Low', line=dict(color='red')))
        fig.update_layout(xaxis_title='Year', yaxis_title='Price', title='Comparing other relevant parameters')
        st.plotly_chart(fig, use_container_width=True)

st.write("#")
#Riskometer
# Create object page
def get_info(url, x):
    score = 0
    page = requests.get(url)
    soup=BeautifulSoup(page.content,'html.parser')
    fund=soup.find_all('td',class_="ft785Value")

    #Fundamental Values
    pb = float(fund[4].text)
    pe = float(fund[2].text)
    de = float(fund[8].text)
    div = float(fund[5].text.replace('%',''))
    roe = float(fund[1].text.replace('%',''))
    indpe = float(fund[6].text)
    pat = soup.find_all('div',class_="shp76TextRight")
    promo = float(pat[0].text.replace('%',''))
    df = get_stats(x)
    l_52 = float(df['Value'][4])
    h_52 = float(df['Value'][3])
    live = round(stock_info.get_live_price(x),2)

    #1 - 52Week
    if abs(live-h_52) <= abs(live-l_52):
        score = score + 1

    #2 - Rev
    if x == 'TCS.NS':
        score = score +1

    #3 - PB
    if pb<3:
        score = score+1
    elif pb>3:
        score = score - 1

    #4 - SHP
    if promo>50:
        score = score+1
    elif promo<50:
        score = score-1

    #5 - Last 5 Year all 3 stocks made profit
    score = score + 1

    #6 - PE
    if pe < 30:
        score = score+1
    elif pe > 100:
        score = score - 1

    #7 - DE
    if de < 1:
        score = score+1
    elif de > 2:
        score = score - 1

    #8 - DivY
    if div > 2:
        score = score + 1

    elif div == 'NULL' or div == 'NA':
        score = score - 1

    #9 - ROE
    if roe > 25:
        score = score + 1
    elif roe < 5:
        score = score - 1

    #10 - Ind
    if pe>indpe:
        score = score-1
    elif abs(pe-indpe)<(indpe*0.1):
        score = score+1
  
    return score

#Access URL object
if comp == 'Tata Consultancy Services - TCS':
    ans = get_info('https://groww.in/stocks/tata-consultancy-services-ltd', 'TCS.NS')
if comp == 'Infosys - INFY':
    ans = get_info('https://groww.in/stocks/infosys-ltd', 'INFY.NS')
if comp == 'Reliance Industries - RELIANCE':
    ans = get_info('https://groww.in/stocks/reliance-industries-ltd', 'RELIANCE.NS')
   
score = 10 - ans

st.subheader('Riskometer')
if score >=9 :
    progress_text = "Very High Risk"
    my_bar = st.progress(0, text=progress_text)

    score = 10 if score>10 else score
    for percent_complete in range(score*10):
        time.sleep(0.02)
        my_bar.progress(percent_complete + 1, text=progress_text)
    st.write(score*10,'%')
    
elif score <=1 :
    progress_text = "Very Low Risk"
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(score*10):
        time.sleep(0.02)
        my_bar.progress(percent_complete + 1, text=progress_text)

    st.write(score*10,'%')
    
elif score <=3 and score>=2:
    progress_text = "Low Risk"
    my_bar = st.progress(40, text=progress_text)

    for percent_complete in range(score*10):
        time.sleep(0.02)
        my_bar.progress(percent_complete + 1, text=progress_text)
    
    st.write(score*10,'%')  
        
elif score <=6 and score >=4 :
    progress_text = "Moderate Risk"
    my_bar = st.progress(60, text=progress_text)

    for percent_complete in range(score*10):
        time.sleep(0.02)
        my_bar.progress(percent_complete + 1, text=progress_text)
    st.write(score*10,'%')
    
elif score <=8 and score >=7 :
    progress_text = "High Risk"
    my_bar = st.progress(80, text=progress_text)

    for percent_complete in range(score*10):
        time.sleep(0.02)
        my_bar.progress(percent_complete + 1, text=progress_text)
    st.write(score*10,'%')

st.caption('Based on 10 fundamental aspects of an equity.')
        
#Predictions
st.write("#")
st.subheader('Predict : ')

if st.button('Click Here'):
    if comp == 'Tata Consultancy Services - TCS':
        x = round(stock_info.get_live_price("TCS.NS"),2)
        tcsweekly = stock_info.get_data("TCS.NS", interval="1d")
        tcsweekly=tcsweekly.dropna()
        values = tcsweekly['close'].values
        data_len = math.ceil(len(values)*0.8) 
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(values.reshape(-1,1))
        test_data = scaled_data[data_len-60: , : ]
        x_test = []
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        new = joblib.load('New/tcsmodelnew.pkl')
        ans = new.predict(x_test)
        ans1 = scaler.inverse_transform(ans)
        val = np.around(ans1[-1][0], decimals=2)
        st.metric(label="Prediction", value=val, delta = round(val-x,2)) 
        
    if comp == 'Reliance Industries - RELIANCE':
        x = round(stock_info.get_live_price("RELIANCE.NS"),2)
        relweekly = stock_info.get_data("RELIANCE.NS", interval="1d")
        relweekly=relweekly.dropna()
        values = relweekly['close'].values
        data_len = math.ceil(len(values)*0.8) 
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(values.reshape(-1,1))
        test_data = scaled_data[data_len-60: , : ]
        x_test = []
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        new = joblib.load('New/relmodelnew.pkl')
        ans = new.predict(x_test)
        ans1 = scaler.inverse_transform(ans)
        val = np.around(ans1[-1][0], decimals=2)
        st.metric(label="Prediction", value=val, delta = round(val-x,2)) 
        
    if comp == 'Infosys - INFY':
        x = round(stock_info.get_live_price("INFY.NS"),2)
        infweekly = stock_info.get_data("INFY.NS", interval="1d")
        infweekly=infweekly.dropna()
        values = infweekly['close'].values
        data_len = math.ceil(len(values)*0.8) 
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(values.reshape(-1,1))
        test_data = scaled_data[data_len-60: , : ]
        x_test = []
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        new = joblib.load('OldModel/infymodelnew.pkl')
        ans = new.predict(x_test)
        ans1 = scaler.inverse_transform(ans)
        val = np.around(ans1[-1][0], decimals=2)
        st.metric(label="Prediction", value=val, delta = round(val-x,2))
        

#Support & Resistance
st.write("#")
st.subheader('Support and Resistance Indicators : ')

def supp_resis(x):
    tcsdaily = stock_info.get_data(x, interval="1d")
    new = tcsdaily.tail(20).head(1)

    high = new['high']
    low = new['low']
    close = new['close']
    pp = (high + low + close)/3
    r1 = 2*pp - low
    s1 = 2*pp - high
    r2 = pp + (r1-s1)
    s2 = pp - (r1-s1)
    r3 = high + 2*(pp-low)
    s3 = low - 2*(high - pp)

    fig = px.line(tcsdaily.tail(20), y='close',markers=False, title=x+' daily data of 1 month')     
    fig.add_hline(y=r1[0], line_dash="dash", line_color="orange",  annotation_text="1st Resistance")
    fig.add_hline(y=s1[0], line_dash="dash", line_color="lime", annotation_text="1st Support")

    fig.add_hline(y=r2[0], line_dash="dash", line_color="red",  annotation_text="2nd Resistance")
    fig.add_hline(y=s2[0], line_dash="dash", line_color="green", annotation_text="2nd Support")

    fig.add_hline(y=r3[0], line_dash="dash", line_color="darkred",  annotation_text="3rd Resistance")
    fig.add_hline(y=s3[0], line_dash="dash", line_color="darkgreen", annotation_text="3rd Support")

    st.plotly_chart(fig, use_container_width=True)

    data = yf.download(
            tickers = x,
            period = "5d",
            interval = "60m",
            group_by = 'ticker',
            auto_adjust = True,
            prepost = False,
            threads = True,
            proxy = None)

    fig = px.line(data, y='Close',markers=False, title=x+' hourly data of 5 days')     
    fig.add_hline(y=r1[0], line_dash="dash", line_color="orange",  annotation_text="1st Resistance")
    fig.add_hline(y=s1[0], line_dash="dash", line_color="lime", annotation_text="1st Support")

    fig.add_hline(y=r2[0], line_dash="dash", line_color="red",  annotation_text="2nd Resistance")
    fig.add_hline(y=s2[0], line_dash="dash", line_color="green", annotation_text="2nd Support")

    fig.add_hline(y=r3[0], line_dash="dash", line_color="darkred",  annotation_text="3rd Resistance")
    fig.add_hline(y=s3[0], line_dash="dash", line_color="darkgreen", annotation_text="3rd Support")

    st.plotly_chart(fig, use_container_width=True)

if comp == 'Tata Consultancy Services - TCS':
    supp_resis('TCS.NS')
if comp == 'Infosys - INFY':
    supp_resis('INFY.NS')
if comp == 'Reliance Industries - RELIANCE':
    supp_resis('RELIANCE.NS')

#Tab for Hist Data
st.write("#")
st.subheader('Financial data : ')
a1, a2, a3 = st.tabs(["Revenue & Profit", "Net Worth", "Shareholding Pattern"])

tier=['Promoters', 'Mutual Funds', 'Retail', 'Foreign Institutions','Others'] 
y=['2018', '2019', '2020', '2021', '2022']

with a1:
    st.caption('All values in Crs')
    if comp == 'Infosys - INFY':
        chart_data = pd.DataFrame([[70522,16029], [82675,15404], [90791,16594], [100472,19351], [121641,22110]],
        index=y, columns=["Revenue", "Profit"])
        st.bar_chart(chart_data, height=350)

    if comp == 'Tata Consultancy Services - TCS':
        chart_data = pd.DataFrame([[123104,25826], [146463,31472], [156949,32430], [164177,32430], [191754,38327]],
        index=y, columns=["Revenue", "Profit"])
        st.bar_chart(chart_data, height=350)

    if comp == 'Reliance Industries - RELIANCE':
        chart_data = pd.DataFrame([[408265,36075], [583094,39588], [611645,39354], [486326,49128], [721634,60705]],
        index=y, columns=["Revenue", "Profit"])
        st.bar_chart(chart_data, height=350)

    
with a2:
    st.caption('All values in Crs')
    if comp == 'Infosys - INFY':
        chart_data = pd.DataFrame([64923, 64948, 65450, 76351, 75350], index=y, columns=['Net Worth'])
        st.bar_chart(chart_data, height=350)
         
    if comp == 'Tata Consultancy Services - TCS':
        chart_data = pd.DataFrame([85128, 89446, 84126, 86433, 89139], index=y, columns=['Net Worth'])
        st.bar_chart(chart_data, height=350)

    if comp == 'Reliance Industries - RELIANCE':
        chart_data = pd.DataFrame([293506, 387112, 453331, 700172, 779485], index=y, columns=['Net Worth'])
        st.bar_chart(chart_data, height=350)

with a3:
    st.caption('As of March, 2023')
    if comp == 'Infosys - INFY':
        x = [15.11, 17.71, 18.22, 36.28, 12.68]
        fig = px.pie(values=x, names=tier)
        st.plotly_chart(fig, use_container_width=True, height=350)

    if comp == 'Tata Consultancy Services - TCS':
        x = [72.30, 3.31, 5.96, 12.94, 5.49]
        fig = px.pie(values=x, names=tier)
        st.plotly_chart(fig, use_container_width=True, height=350)

    if comp == 'Reliance Industries - RELIANCE':
        x = [50.49, 5.81, 11.64, 23.43, 8.63]
        fig = px.pie(values=x, names=tier)
        st.plotly_chart(fig, use_container_width=True, height=350)

st.write('Thanks ! We hope our webpage was useful.')
st.caption('The Web Application was made by Anand Soni and Deepak Rathore.')
