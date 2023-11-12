import joblib
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
from streamlit_option_menu import option_menu
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def registered_app():
    st.header('Prediction for Registered Users')
    st.markdown('---')
    text = 'This  is a specialized module designed to predict the demand for registered users'
    st.markdown(f'<p style="text-align: justify;">{text}</p>', unsafe_allow_html=True)
    model_dem = joblib.load('model_reg_final.pkl')
    temp = st.slider('Temperature at time of prediction', min_value=0, max_value=50, value=4, step=1)
    hum = st.slider('Humidity at time of prediction', min_value=0, max_value=100, value=40, step=1)
    MovingAvg_Reg = st.number_input('What is the moving average demand of registered users?')
    lag_1_Reg = st.number_input('Demand of Registered Riders prior hour')
    lag_24_reg = st.number_input('Demand of Registered Riders day before (24 hours earlier)')
    lag_week_reg = st.number_input('Demand of Registered Riders week before (7 days ago at same hour)')
    #weekday = st.radio('What day of the week is it?', ['Monday', 'Tuesday', "Wednesday","Thursday", "Friday","Saturday","Sunday"])
    workingday = st.radio('Is selected day working day?', ['Yes', 'No'])
    time_slot = st.slider('Time of Day', min_value=0, max_value=23, value=8, step=1)
    season = st.radio('What season is it?', ['Winter', 'Spring',"Summer","Fall"])
    weathersit = st.radio('What is the weather like?', ['Clear / Few Clouds', 'Misty',"Light Rain / Snow","Heavy Rain / Snow"])

    if st.button('Predict Demand'):
        print("Button Demand")
        # Convert categorical values to numerical
        hour_sin = np.sin(2 * np.pi * time_slot / 12.0)
        hour_cos = np.cos(2 * np.pi * time_slot / 12.0)
        workingday_1 = 1 if workingday == 'Yes' else 0
        workingday_0 = 1 if workingday == "No" else 0
        time_slot_afternoon_rush_hour = 1 if time_slot in range(16,20) else 0
        time_slot_morning_rush_hour = 1 if time_slot in range(7,11) else 0
        weathersit_3 = 1 if weathersit == "Light Rain / Snow" else 0
        season_1 = 1 if season == "Winter" else 0
        time_slot_night = 1 if time_slot in range(20,24) else 0

        data_dem = {
            'temperature': temp,
            'humidity': hum/100,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'MovingAvg_Reg': MovingAvg_Reg,
            'lag_1_Reg': lag_1_Reg,
            'lag_24_reg': lag_24_reg,
            'lag_week_reg': lag_week_reg,
            'season_1':season_1, 
            'workingday_0': workingday_0,
            'workingday_1': workingday_1,
            'weathersit_3': weathersit_3,
            'time_slot_afternoon_rush_hour': time_slot_afternoon_rush_hour,
            'time_slot_morning_rush_hour': time_slot_morning_rush_hour,
            'time_slot_night': time_slot_night
        }


       
        df_dem = pd.DataFrame.from_dict([data_dem])
        scaler_dem = joblib.load('scaler_reg_final.pkl')
        data_dem = scaler_dem.transform(df_dem)
        # changing the input_data to numpy array
        input_datadem_as_numpy_array = np.asarray(data_dem)

        # reshape the array as we are predicting for one instance
        input_datadem_reshaped = input_datadem_as_numpy_array.reshape(1, -1)

        prediction_dem = model_dem.predict(input_datadem_reshaped)

        rounded_predictions = np.ceil(prediction_dem)[0]


        # ...

        # Mostrar el resultado redondeado
        st.success('Demand Forecast by Regular Users is {}'.format(rounded_predictions))    
        

        #st.success('Demand Forecast by Regular Users is {:.2f}'.format(round(prediction_dem,2)))
        
        st.title('Model Development - Registered Demand Prediction')
        df = pd.read_csv('bike-sharing_hourly.csv')

        

        # daily seasonality
        df['hour_sin'] = np.sin(2 * np.pi * df['hr'] / 12.0) 
        df['hour_cos'] = np.cos(2 * np.pi * df['hr'] / 12.0)

        # moving average of last 7 observations
        df['MovingAvg_Reg']=df["registered"].rolling(window=7).mean()

        # Lag 1h
        df['lag_1_Reg']=df["registered"].shift(1)
        #Lag 24h
        df['lag_24_reg']=df["registered"].shift(24)

        #weekly seasonality
        df['lag_week_reg']=df["registered"].shift(24*7) 

        df= df.drop(["cnt","casual","atemp","instant","dteday","yr","mnth"], axis=1)

        X_corr= df.drop(["hr","registered"], axis=1)
        X_corr.dropna(inplace=True) 

        train_portion=0.8
        train_set_size=int(train_portion*len(X_corr))
        
        X_train_reg_corr=X_corr[:train_set_size]
        
        
        st.write('Review Correlation between Features of Casual Model ')

        # Calculate the correlation matrix
        correlations = X_train_reg_corr.corr()

        plt.figure(figsize=(15, 15))
        fig, ax = plt.subplots()
        sns.heatmap(correlations, annot=True, cmap='coolwarm', linewidths=0.1)
        plt.subplots_adjust(left=0, right=2, bottom=0, top=2, wspace=0.2, hspace=0.5)

        
        # Display the heatmap figure
        st.pyplot(fig)
        st.write('We notice that temperature and lags are particulalry are particulalry correlated (which makes sense)')

        #Create Time slots (could not create earlier because correlation table cannot handle this column in streamlit)
        # Create time slots
        bins = [0, 6, 10, 15, 19, 24] # Define the bin edges for the hours (in military time)
        labels = ['night', "morning_rush_hour", 'mid_day', 'afternoon_rush_hour', 'evening'] # Define labels for the categories
        df['time_slot'] = pd.cut(df['hr'], bins=bins, labels=labels, include_lowest=True)

        y_regist = df['registered']
        y_reg = df['registered']

        X= df.drop(["hr","registered"], axis=1)
        X = X.dropna()

        

        st.image("F-Score_registered.jpg")

        st.write('Using the KBest Algorithim, this graph clearly shows us that bike demand from registered users is most impacted by demand in the hour before (lag_1_casual), demand of registered users from the day before, and the moving average of registered demand. Another Important feature are the rush hour timeslots, weather is notable much less important than for casual riders, which aligns with our EDA')

        st.write("Over all the 15 most important features that were selected for the Casual Model are:\n\n"
                    'lag_1_reg,\n\n'
                    'lag_24_reg,\n\n'
                    'MovingAvg_Reg,\n\n'
                    'time_slot_night, \n\n' 
                    'time_slot_afternoon_rush_hour,\n\n'
                    'lag_week_reg,\n\n'
                    'temp, \n\n'
                    'hour_cos, \n\n'
                    'hum,\n\n'
                    'hour_sin,\n\n'
                    'season_1,\n\n'
                    'time_slot_morning_rush_hour,\n\n'
                    'workingday_0,\n\n'
                    'workingday_1,\n\n'
                    'weathersit_3, \n\n')
        st.header('Choosing a Model')
        st.write('By using RandomizedSearchCV, we explore various algorithms and their hyperparameters to find the best fit. We are particularly focused on minimizing the Mean Absolute Error (MAE) it was choose as  metric that penalizes underestimation more than overestimation might be appropriate, as the bike-sharing company has a high cost for not meeting the demand and the cost of overestimating  is also significant.')


        st.header('Model Output Analysis')
        st.image('Results_reg.jpg')
        st.write('We tested our model using a 80:20 Train/Test Split. Since this is a time series, we did this by ommitting the last 20 percent of the data set for training purposes\n\n'
            'The ommitted data was then used for training purposes. The output of the training data versus the output can be observed below')
        st.image('Model_Forecast_reg.jpg')
                

        