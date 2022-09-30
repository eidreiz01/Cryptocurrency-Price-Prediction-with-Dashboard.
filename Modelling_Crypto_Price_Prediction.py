#Run this code in a Conda Environment
###Note to run ---> "pip install pystan" in terminal
###Note to run ---> "conda install -c conda-forge fbprophet" in terminal

##Importing necessary libraries

#Used for calculation
import numpy as np
#Used for data manipulation
import pandas as pd
#used for plotting visualizations
import matplotlib.pyplot as plt
#used for retrieving cryptocurrency data
import yfinance as yf
#used for plotting visualizations
import plotly.express as px
#used for time related functions
import time
#used for creating dashboards
import streamlit as st
#used for converting time formats
import datetime
#used for plotting visualizations
import seaborn as sns
#used for creating machine learning model
import fbprophet as fb
from prophet import Prophet


#a list containing tickers for different cryptocurrencies (20 in total)
tickers_list = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'USDC-USD', 'HEX-USD',
 'XRP-USD',
 'LUNA1-USD',
 'SOL-USD',
 'ADA-USD',
 'UST-USD',
 'BUSD-USD',
 'DOGE-USD',
 'AVAX-USD',
 'DOT-USD',
 'SHIB-USD',
 'WBTC-USD',
 'STETH-USD',
 'DAI-USD',
 'MATIC-USD']

#a list containing different durations available to the user for prediction
duration_type_list = ["day", "week", "month", "quarter"]
#a map mapping the different durations to the number of days
duration_map = {"day":1, "week":7, "month":30, "quarter":90}
#a list containing different machine learning models available to the user for prediction
models_list = ["fb-prophet"]

#a method called by streamlit to set dashboard page title, icon and type of layout (all this info is present under inspection tab in the web browsers)
st.set_page_config(
    page_title = 'Real-Time Data Science Dashboard',
    page_icon = '✅',
    layout = 'wide'
)

#a method called by streamlit to set dashboard title (this title is the one shown on dashboard)
st.title("SOLiGence Real-Time Cryptocurrency Dashboard")

#selectbox is used to take input from a user in the form of choices, this selectbox presents user with choice of cryptocurrencies
currency_filter = st.selectbox("Select the Currency", tickers_list)

#this function is used to train the fbprophet machine learning model
def fbprophet_model(data, currency_filter, data_prediction):
  #instantiating the fbprophet model object
  model_ = fb.Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
  #creating a new dataframe for fbprophet specific input
  data_train = pd.DataFrame()
  #create two columns which are required for fbprophet specfic input
  data_train['ds'] = data.reset_index()['Date']
  data_train['y'] = data.reset_index()[currency_filter]['Close']
  #removing null values from the dataset
  data_train.dropna(inplace=True)
  #training fbprophet model
  model_.fit(data_train)
  #creating fbprophet specfic dataframe for prediction
  data_pred = pd.DataFrame(data_prediction, columns=['ds'])
  #returning predictions from fbprophet model
  return model_.predict(data_pred)['yhat']

#this function is used to find dates within the input interval with predicted profit more than the required profit
def find_profit(data, times, forecast, profit, currency):
  #find the latest closing value of selected currency
  current_value = data[currency]['Close'][-1]
  #create the output dataframe
  df_ = pd.DataFrame(columns=["Currency", "Date", "Expected Profit"])
  #appending the rows in the dataframe based on profit condition
  for i in range(len(forecast)):
    if (forecast[i]-current_value) >= profit:
      df_.loc[len(df_.index)] = [currency, times[i], forecast[i]-current_value]
  #returning output dataframe and latest crypto closing value
  return df_, current_value

#this function creates the future date values in interval of days for the required input duration
def get_prediction_data(data, duration):
  #store latest date from the downloaded data
  current_date = data.index[-1]
  #create a list of dates starting from the latest date till required duration with interval of 1 day
  times = [(current_date.to_pydatetime() + datetime.timedelta(days=x)) for x in range(duration_map[duration])]
  #return teh interval dates list
  return times

#creating an empty container object using the streamlit library
placeholder = st.empty()

#create a container using the placeholder object
with placeholder.container():
	#download the data using yfinance library for required crypto tickers
    df = yf.download(tickers_list, period="max", interval="1d", group_by='tickers')
    #since downloaded data is a multi-level column dataframe grouped by crypto tickers, a pandas slice object is used to select only closing values for all the tickers
    idx = pd.IndexSlice
    df_tmp = df.loc[:,idx[:,'Close']]
    #create a row containing two columns and store the columns container as variables
    fig_col1, fig_col2 = st.columns(2)
    #set required data in the first column of the first row
    with fig_col1:
    	#set title for the first column of the first row
        st.markdown("### First Chart")
        #creating the line figure for opening values
        fig = px.line(df[currency_filter], y="Open")
        st.write(fig)
    #set required data in the second column of the first row
    with fig_col2:
    	#set title for the second column of the first row
        st.markdown("### Second Chart")
        #creating the line figure for closing values
        fig2 = px.line(df[currency_filter], y="Close")
        st.write(fig2)
    #create a row containing two columns and store the columns container as variables
    fig_col3, fig_col4 = st.columns(2)
    #set required data in the first column of the second row
    with fig_col3:
    	#set title for the first column of the second row
        st.markdown("### Third Chart")
        #creating the line figure for high values
        fig3 = px.line(df[currency_filter], y="High")
        st.write(fig3)
    #set required data in the second column of the second row
    with fig_col4:
    	#set title for the second column of the second row
        st.markdown("### Fourth Chart")
        #creating the line figure for low values
        fig4 = px.line(df[currency_filter], y="Low")
        st.write(fig4)
    #create a row containing two columns and store the columns container as variables
    fig_col5, fig_col6 = st.columns(2)
    #set required data in the first column of the third row
    with fig_col5:
    	#set title for the first column of the third row
        st.markdown("### Fifth Chart")
        #creating the line figure for adjusted close values
        fig5 = px.line(df[currency_filter], y="Adj Close")
        st.write(fig5)
    #set required data in the second column of the third row
    with fig_col6:
    	#set title for the second column of the third row
        st.markdown("### Sixth Chart")
        #creating the line figure for volume values
        fig6 = px.line(df[currency_filter], y="Volume")
        st.write(fig6)
    #create a row containing two columns and store the columns container as variables
    fig_col7, fig_col8 = st.columns(2)
    #set required data in the first column of the fourth row
    with fig_col7:
    	#set title for the first column of the fourth row
        st.markdown("### Seventh Chart")
        fig7 = plt.figure()
        #creating the heatmap figure for correlation values
        sns.heatmap(df_tmp.corr())
        st.write(fig7)
    #set required data in the second column of the fourth row
    with fig_col8:
    	#set title for the second column of the fourth row
        st.markdown("### Eigth Chart")
        #creating a dummy dataframe for calculating the rolling mean for closing values
        df_copy = pd.DataFrame()
        df_copy['Close'] = df[currency_filter]['Close']
        df_copy = df_copy.reset_index()
        #calculate the rolling mean for 7 days
        df_copy['rolling_mean'] = df_copy['Close'].rolling(7).mean()
        fig8 = plt.figure()
        #creating the lineplot figure for closing values
        sns.lineplot( x = 'Date',
                    y = 'Close',
                    data = df_copy,
                    label = 'Close Values')
        #creating the lineplot figure for rolling mean of closing values
        sns.lineplot( x = 'Date',
                    y = 'rolling_mean',
                    data = df_copy,
                    label = 'Rolling Close Values')
        st.write(fig8)
    #create an expander container usign streamlit library for finding relevant crypto predictions
    with st.expander("Find your bet!!"):
      #create a select box for duration types
      duration_filter = st.selectbox("Select the Duration", duration_type_list)
      #create an input field for profit
      profit_filter = int(st.number_input('Insert the required profit'))
      #create a select box for available machine learning models
      model_filter = st.selectbox("Select the Predictor Model", models_list)
      #create a submit button for input fields inside the expander container and output the results
      if st.button('Submit'):
      	#create a spinner to show progress while calculation is carried out
        with st.spinner('In progress...'):
        	#call the get_prediction_data function to get the list of dates with 1 day intervals
            pred_data = get_prediction_data(df, duration_filter)
            #use appropriate machine learning model based on the input given by the user
            if model_filter=="fb-prophet":
              #training the fbprophet model
              forecast = fbprophet_model(df, currency_filter, pred_data)
            #store the output dataframe and the latest crypto close value
            output, current_value = find_profit(df, pred_data, forecast, profit_filter, currency_filter)
            #print the output message
            st.write("Output Dataframe is generated!! with current value as " + str(current_value))
            #print the forecasts as string
            for i in range(len(pred_data)):
              st.write("" + str(pred_data[i]) + "\t\t" + str(forecast[i]))
            #print the forecasts as a dataframe
            st.dataframe(output)
            #if output dataframe is not empty, then plot a bar chart containing the expected profits
            if output.shape[0]!=0:
              #create the expected profits bar plot
              fig3 = px.bar(output, x="Date", y="Expected Profit")
              st.write(fig3)
            else:
              #if output dataframe is empty meaning it contains no rows, then tell the user that no required date is found
              st.write("No dates found!!")
