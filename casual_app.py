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
import gdown
import os

def download_model(file_id, output_path):
    # Construct the gdown URL
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # Download the file
    gdown.download(url, output_path, quiet=False)

# The ID of your file on Google Drive (just the ID, not the full URL)
FILE_ID = '1IXqQhGg_s8yF6t-ZQLDAkPXCWgd0ioO2' 

# Check if the file does not already exist to avoid re-downloading it
if not os.path.exists('model_casual_final.pkl'):
    # Call the download function with the actual FILE_ID and the desired output path
    download_model(FILE_ID, 'model_casual_final.pkl')

# Now the model file should be in your local directory and can be loaded
with open('model_casual_final.pkl', 'rb') as ft_file:
    model_dem = pickle.load(ft_file)


#https://drive.google.com/file/d/1IXqQhGg_s8yF6t-ZQLDAkPXCWgd0ioO2/view?usp=sharing

def casual_app():
    st.header('Prediction for Casual Users')
    st.markdown('---')
    text = 'This  is a specialized module designed to predict the demand for casual users'
    st.markdown(f'<p style="text-align: justify;">{text}</p>', unsafe_allow_html=True)
    #model_dem = joblib.load('model_casual_final.pkl')
    temp = st.slider('Temperature at time of prediction', min_value=0, max_value=50, value=4, step=1)
    hum = st.slider('Humidity (in percentage) at time of prediction', min_value=0, max_value=100, value=40, step=1)
    MovingAvg_casual = st.number_input('What is the moving average demand of casual users?')
    lag_1_casual = st.number_input('Demand of Casual Riders prior hour')
    lag_24_casual = st.number_input('Demand of Casual Riders day before (24 hours earlier)')
    lag_week_casual = st.number_input('Demand of Casual Riders week before (7 days ago at same hour)')
    weekday = st.radio('What day of the week is it?', ['Monday', 'Tuesday', "Wednesday","Thursday", "Friday","Saturday","Sunday"])
    workingday = st.radio('Is selected day working day?', ['Yes', 'No'])
    time_slot = st.slider('Time of Day', min_value=0, max_value=23, value=8, step=1)
    season = st.radio('What season is it?', ['Winter', 'Spring',"Summer","Fall"])

    if st.button('Predict Demand'):
        print("Button Demand")
        # Convert categorical values to numerical
        workingday_1 = 1 if workingday == 'Yes' else 0
        workingday_0 = 1 if workingday == "No" else 0
        time_slot_afternoon_rush_hour = 1 if time_slot in range(16,20) else 0
        time_slot_mid_day = 1 if time_slot in range(11,16) else 0
        time_slot_night = 1 if time_slot in range(20,24) else 0
        weekday_0 = 1 if weekday == "Sunday" else 0
        weekday_6 = 1 if weekday == "Saturday" else 0
        season_1 = 1 if season == "Winter" else 0
        season_2 = 1 if season == "Spring" else 0

        data_dem = {
            'temperature': temp,
            'humidity': hum/100,
            'MovingAvg_casual': MovingAvg_casual,
            'lag_1_casual': lag_1_casual,
            'lag_24_casual': lag_24_casual,
            'lag_week_casual': lag_week_casual,
            'season_1':season_1,
            'season_2':season_2, 
            'weekday_0': weekday_0,
            'weekday_6': weekday_6,
            'workingday_0': workingday_0,
            'workingday_1': workingday_1,
            'time_slot_afternoon_rush_hour': time_slot_afternoon_rush_hour,
            'time_slot_mid_day': time_slot_mid_day,
            'time_slot_night': time_slot_night
        }


       
        df_dem = pd.DataFrame.from_dict([data_dem])
        scaler_dem = joblib.load('scaler_casual_final.pkl')
        data_dem = scaler_dem.transform(df_dem)
        # changing the input_data to numpy array
        input_datadem_as_numpy_array = np.asarray(data_dem)

        # reshape the array as we are predicting for one instance
        input_datadem_reshaped = input_datadem_as_numpy_array.reshape(1, -1)

        prediction_dem = model_dem.predict(input_datadem_reshaped)
        rounded_predictions = np.ceil(prediction_dem)[0]

        st.success('Demand Forecast by Casual Users is {}'.format(rounded_predictions))    
        

        #st.success('Demand Forecast by Regular Users is {:.2f}'.format(round(pre
        
        st.title('Model Development - Casual Demand Prediction')
        df = pd.read_csv('bike-sharing_hourly.csv')

        # daily seasonality
        df['hour_sin'] = np.sin(2 * np.pi * df['hr'] / 12.0) 
        df['hour_cos'] = np.cos(2 * np.pi * df['hr'] / 12.0)

        # moving average of last 7 observations
        df['MovingAvg_casual']=df["casual"].rolling(window=7).mean()

        # Lag 1h
        df['lag_1_casual']=df["casual"].shift(1)
        #Lag 24h
        df['lag_24_casual']=df["casual"].shift(24)

        #weekly seasonality
        df['lag_week_casual']=df["casual"].shift(24*7) 


        # drop the rows that are NA created through the lagging as the dataset big enough 
        df.dropna(inplace=True) 
        df =df.drop("dteday",axis=1)


        # Set up data split into target and feature
        from sklearn.feature_selection import SelectKBest, f_regression
        X= df.drop(['cnt','casual','registered', "hr", "yr","instant", "mnth", "atemp"],axis=1)
        y_casual = df['casual']
        y_regist = df['registered']

        # Train / Test split 
        train_portion=0.8
        train_set_size=int(train_portion*len(X))
        X_train_casual=X[:train_set_size]
        y_train_casual=y_casual[:train_set_size]


        X_test_casual=X[train_set_size:]
        y_test_casual=y_casual[train_set_size:]

        st.write('Review Correlation between Features of Casual Model ')

        # Calculate the correlation matrix
        correlations = X_train_casual.corr()

        plt.figure(figsize=(15, 15))
        fig, ax = plt.subplots()
        sns.heatmap(correlations, annot=True, cmap='coolwarm', linewidths=0.1)
        plt.subplots_adjust(left=0, right=2, bottom=0, top=2, wspace=0.2, hspace=0.5)

        
        # Display the heatmap figure
        st.pyplot(fig)


        # Create Time slots (could not create earlier because correlation table cannot handle this column)
        # Create time slots
        bins = [0, 6, 10, 15, 19, 24] # Define the bin edges for the hours (in military time)
        labels = ['night', "morning_rush_hour", 'mid_day', 'afternoon_rush_hour', 'evening'] # Define labels for the categories
        df['time_slot'] = pd.cut(df['hr'], bins=bins, labels=labels, include_lowest=True)

        st.write('We notice that the lags as well as temperature are highly correlated to each other, which is intutive and align with our EDA')

        # OneHotEncode categorcial columns for K-Best Algorithim

        X_cat = X_train_casual.select_dtypes(include=["int", "category"])

        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        ohe = OneHotEncoder(sparse_output=False)
        cat_data_ohe = ohe.fit_transform(X_cat)
        cat_data_ohe = pd.DataFrame(cat_data_ohe, columns=ohe.get_feature_names_out())
        X_full_casual = pd.concat([X_train_casual.reset_index(drop=True), cat_data_ohe], axis=1)
        X_full_casual = X_full_casual.drop(columns=X_cat.columns)

        sel_casual = SelectKBest(score_func=f_regression, k=15)  
        sel_casual.fit(X_full_casual, y_train_casual)

        scores = pd.Series(sel_casual.scores_, index=X_full_casual.columns)
        scores = scores.sort_values(ascending=False)
        fig = px.bar(scores, template="none", title="F-Score of features with casual as dependent variable")

        # Display the figure in Streamlit
        st.plotly_chart(fig)

        st.write('Using the KBest Algorithim, this graph clearly shows us that bike demand from casual users is most impacted by demand in the hour before (lag_1_casual), the moving average of casual demand, and demand from the day before. Another Important feature is the temperature, which aligns with our EDA')

        st.write("Over all the 15 most important features that were selected for the Casual Model are:\n\n"
            'Temperature,\n\n'
            'hum,\n\n'
            'MovingAvg_casual,\n\n'
            'lag_1_casual,\n\n'
            'lag_24_casual,\n\n'
            'lag_week_casual,\n\n' 
            'season_1,\n\n'
            'season_2,\n\n'
            'weekday_0,\n\n'
            'weekday_6,\n\n'
            'workingday_0,\n\n'
            'workingday_1,\n\n'
            'time_slot_afternoon_rush_hour,\n\n'
            'time_slot_mid_day,\n\n'
            'time_slot_night')

        st.header('Choosing a Model')
        st.write('By using RandomizedSearchCV, we explore various algorithms and their hyperparameters to find the best fit. We are particularly focused on minimizing the Mean Absolute Error (MAE) it was choose as  metric that penalizes underestimation more than overestimation might be appropriate, as the bike-sharing company has a high cost for not meeting the demand and the cost of overestimating  is also significant.')


        st.header('Model Output Analysis')
        st.image('Results.jpg')
        st.write('We tested our model using a 80:20 Train/Test Split. Since this is a time series, we did this by ommitting the last 20 percent of the data set for training purposes\n\n'
            'The ommitted data was then used for training purposes. The output of the training data versus the output can be observed below')
    
        st.image('Model_Forecast.jpg')
