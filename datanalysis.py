import joblib
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from io import StringIO


def data_analysis_app():
    st.markdown("<h1 style='text-align: center;'>Data Analysis</h1>", unsafe_allow_html=True)
    
    texto = ('The soundness of a powerful model is highly dependent on a strong Exploratory Data Analysis (EDA). The following section will outline how the team went about ensuring data quality')
    st.markdown(f'<p style="text-align: justify;">{texto}</p>', unsafe_allow_html=True)

    st.subheader('Data Quality Review')
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv('bike-sharing_hourly.csv')

    buffer = StringIO()
    # Redirect the output of df.info() to buffer
    df.info(buf=buffer)
    # Get the content from buffer
    s = buffer.getvalue()

    # Use st.text to display the info
    st.text(s)
    st.write('After an initial review of the data structure, it was decided to drop the column Instant because it is the same as index. Also we will drop the yr and dteday column, as the year should not impact our model and all other date related information can be inferred from the remaining columns.')

    # Drop the instant column, since it is repetitive of the index as well as the year column
    df.drop(['instant',"dteday", "yr"], axis=1, inplace=True)


    # Check for outliers in the numeric columns

    st.write('We checked for outliers in the numeric columns and noticed the following columns to have a high number of outliers (particularly in the wind column), which decided to clean using 1.5 IQR to ensure the model quality')

    # BoxPlots for outliers
    # Selecting numerical columns of type 'float64'
    numerical_cols = df.select_dtypes(include='float64').columns

    # Creating subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=numerical_cols)

    # Adding box plots to each subplot
    for i, col in enumerate(numerical_cols, 1):
        row = (i - 1) // 2 + 1
        col_pos = (i - 1) % 2 + 1
        fig.add_trace(go.Box(y=df[col], name=col), row=row, col=col_pos)

    # Updating layout
    fig.update_layout(height=300 * 2, width=900, title_text="Box Plots for Outlier Analysis")

    # Displaying the figure in Streamlit
    st.plotly_chart(fig)




    # Calculate the correlation matrix
    correlations = df.corr()

    # Display the correlation matrix as a heatmap
    fig = go.Figure(data=go.Heatmap(
    z=correlations,
    x=correlations.columns,
    y=correlations.columns,
    ))

    # Adding annotations with increased font size
    for i, row in enumerate(correlations.values):
        for j, value in enumerate(row):
            fig.add_annotation(dict(
                font=dict(size=12),
                x=correlations.columns[j],
                y=correlations.columns[i],
                text="{:.2f}".format(value),
                showarrow=False,
                xref="x",
                yref="y"))

    # Update the layout
    fig.update_layout(title='Correlation Analysis',
                      xaxis=dict(tickangle=-30),
                      margin=dict(l=20, r=20, t=40, b=20))

    # Display the heatmap in Streamlit
    
    st.plotly_chart(fig)

    st.write('**Findings**: According to the results in the correlation matrix, we can notice that there is a high correlation between independent variables, for instance temp and atemp variables (corr=0.99), which make sense because both are related to temperature, one is the actual temperature and the other one is the feeling.\n\n'
        'We can drop the atemp column given high correlation with temp column. Also, there is a high correlation among cnt, registered and casual, which also make sense because the variable cnt is the total between registered and casual. We conclude that for the business use case, a registered and a casual model would be better.\n\n' 
        'Esepcially casual riders are quite correlated with temperature, which makes sense since no one would opt to purchase a bike ride when its cold')

    st.title('Data Visualization')

    # Get the unique days of the week
    days = sorted(set(df['weekday']))

    # Calculate the total number of casual and registered riders for each day of the week
    casual_driver_per_week = df.groupby('weekday')['casual'].sum()
    registered_driver_per_week = df.groupby('weekday')['registered'].sum()

    fig = go.Figure()

    # Create a bar chart
    fig.add_trace(go.Bar(
    x=days,
    y=casual_driver_per_week,
    name='Casual Clients',
    marker_color='blue'))

    fig.add_trace(go.Bar(
        x=[day + 0.35 for day in days],
        y=registered_driver_per_week,
        name='Registered Clients',
        marker_color='red'))

    # Customize the chart
    fig.update_layout(
        title='Casual vs Registered Clients per Day of Week',
        xaxis_title='Day of the Week',
        yaxis_title='Number of Clients',
        xaxis=dict(tickvals=[day + 0.35 for day in days], ticktext=days),
        legend=dict(x=0, y=1.0),
        barmode='group')

    # Display the chart in Streamlit
    st.plotly_chart(fig)

    st.write('Observation: This view confirms our idea to create seperate models, since we observe different trends in usage based on Casual and Registered Clients. We notice that on weekends, bicycle use increases for casual clients, whereas registered clients tend to bike in greater numbers during the week than on the weekend. Intuitvely, this makes sense. Registered users are likely use the bikes for commuting, wherease casual riders (including tourists, etc) for leisure during weekedn visits.')

    df = pd.read_csv('bike-sharing_hourly.csv')

    # Get the unique hours of the day
    hours = sorted(set(df['hr']))

    # Calculate the total number of casual and registered riders for each hour of the day
    casual_driver_per_hour = df.groupby('hr')['casual'].sum()
    registered_driver_per_hour = df.groupby('hr')['registered'].sum()

    # Create a bar chart using Plotly
    fig = go.Figure()

    # Add bar plots for casual and registered drivers
    fig.add_trace(go.Bar(
        x=hours,
        y=casual_driver_per_hour,
        name='Casual Clients',
        marker_color='blue'  # You can change the color as needed
    ))

    fig.add_trace(go.Bar(
        x=[hour + 0.35 for hour in hours],
        y=registered_driver_per_hour,
        name='Registered Clients',
        marker_color='red'  # You can change the color as needed
    ))

    # Customize the chart
    fig.update_layout(
        title='Casual vs Registered Clients per Time of Day',
        xaxis_title='Time of Day',
        yaxis_title='Number of Clients',
        xaxis=dict(tickvals=[hour + 0.35 for hour in hours], ticktext=hours),
        legend=dict(x=0, y=1.0),
        barmode='group'
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)
    st.write('This graph shows that the use of the service for bikes has certain peak hours. However, the timing of the peak differs by user. Registered peak hours occur during rush hour, namely between 7-9am and between 5 and 7pm, which underscores our belief that registered users use the bikes for commuting. Usage rates of casual users on the other hand only gradually increase throughout the day, peaking between 1-5pm. With that in mind, we decide to create 5 buckets to capture time of day, as using the individual hour in our dataset would likely result in model overfit')

    alt_df = df
    # Create Time slots (could not create earlier because correlation table cannot handle this column)
    # Create time slots
    bins = [0, 6, 10, 15, 19, 24]  # Define the bin edges for the hours (in military time)
    labels = ['night', "morning_rush_hour", 'mid_day', 'afternoon_rush_hour', 'evening']  # Define labels for the categories
    alt_df['time_slot'] = pd.cut(alt_df['hr'], bins=bins, labels=labels, include_lowest=True)

    # Calculate the total number of casual and registered riders for each time slot
    casual_driver_per_slot = alt_df.groupby('time_slot')['casual'].sum()
    registered_driver_per_slot = alt_df.groupby('time_slot')['registered'].sum()

    # Create a bar chart using Plotly
    fig = go.Figure()

    # Add bar plots for casual and registered drivers
    fig.add_trace(go.Bar(
        x=casual_driver_per_slot.index,  # Time slots as x-axis
        y=casual_driver_per_slot,
        name='Casual Clients',
        marker_color='blue'  # You can change the color as needed
    ))

    fig.add_trace(go.Bar(
        x=registered_driver_per_slot.index,  # Time slots as x-axis
        y=registered_driver_per_slot,
        name='Registered Clients',
        marker_color='red'  # You can change the color as needed
    ))

    # Customize the chart
    fig.update_layout(
        title='Casual vs Registered Clients per Time Slot',
        xaxis_title='Time Slot',
        yaxis_title='Number of Clients',
        legend=dict(x=0, y=1.0),
        barmode='group'
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig)



    # Group the data by day and working day and sum the number of casual riders
    total_casual = df.groupby(['dteday','workingday'])['casual'].sum().reset_index()

    # Create a line chart with Plotly Express
    fig = px.line(total_casual, x='dteday', y='casual', title='Total bikes rented by casual users',color='workingday')

    # Display the chart in Streamlit
    st.plotly_chart(fig)

    st.write('In this time series, it is observed that casual bicycle users use the service more frequently on non-working days, a factor that could be related to the availability of free time and the preference for recreational rides during non-working days.')

    # Group the data by day and working day and sum the number of registered riders
    total_registered = df.groupby(['dteday','workingday'])['registered'].sum().reset_index()

    # Create a line chart with Plotly Express
    fig = px.line(total_registered, x='dteday', y='registered', title='Total bikes rented by registered users',color='workingday')

    # Display the chart in Streamlit
    st.plotly_chart(fig)

    st.write('In this time series, it is observed that registered bicycle users use of the fluctuates significantly, but is generally higher on working days - which stregnthens our assumption that registered user use the big for commuting. We also observe some seasonal fluctations, as usage rates signifincantly decrease in winter months. Potential outliers, such as Oct. 29 2012 we reviewed and deemed accurate, as this drop is related to Hurrciane Sandy.')


    # Create a figure with Plotly Graph Objects
    fig = go.Figure()

    # Add a trace for the registered users
    fig.add_trace(go.Scatter(
        x=total_registered.dteday,
        y=total_registered.registered,
        mode='lines',
        name='Registered users'
    ))

    # Add a trace for the casual users
    fig.add_trace(go.Scatter(
        x=total_casual.dteday,
        y=total_casual.casual,
        mode='lines',
        name='Casual users'
    ))

    # Update the x-axis title
    fig.update_xaxes(title_text="Date")

    # Update the y-axis title
    fig.update_yaxes(title_text="Total bikes")

    # Update the layout title
    fig.update_layout(title='Total bikes of registered users and casual users')

    # Display the figure in Streamlit
    st.plotly_chart(fig)

    st.write('The time series consistently reveals that over time, registered users show significantly higher demand compared to casual users of the bike service. This persistent pattern could be related to registered users loyalty to the service and their dependence on bicycles as a regular transportation option to carry out their daily activities.')

    # Group the data by temperature and sum the number of riders
    casual_temp = df.groupby(['temp'])['casual'].sum().reset_index()
    casual_temp['temp'] = (casual_temp['temp'] * 41)
    reg_temp = df.groupby(['temp'])['registered'].sum().reset_index()
    reg_temp['temp'] = (reg_temp['temp'] * 41)

    # Create an empty figure
    fig = go.Figure()

    # Add the line for casual riders
    fig.add_trace(go.Scatter(
        x=casual_temp.temp,
        y=casual_temp.casual,
        mode='lines',
        name='Casual users'
    ))

    # Add the line for registered riders
    fig.add_trace(go.Scatter(
        x=reg_temp.temp,
        y=reg_temp.registered,
        mode='lines',
        name='Registered users'
    ))

    # Update the x-axis title
    fig.update_xaxes(title_text="Temperature in degrees Celcius")

    # Update the y-axis title
    fig.update_yaxes(title_text="Usage")

    # Display the figure in Streamlit
    st.plotly_chart(fig)

    st.write('For this graph we multiply the temperature by 41 with the aim of obtaining values that are more understandable to the public in our visualizations. It is observed that the demand for the bicycle service is greater at times in which the temperature is between 25 and 30 degrees Celcius, and its demand decreases when the temperature reaches extreme values (between 0 and 7 degrees Celcius and greater than 37 degrees Celcius).')

    # Create a dictionary to map season numbers to season names
    season_names = {1:'Winter', 2:'Spring', 3:'Summer', 4:'Fall'}

    # Calculate the total number of bikes rented by registered and casual users for each season
    total_season = df.groupby(['season'])[['casual', 'registered']].sum().reset_index()
    total_season['season_name'] = total_season['season'].map(season_names)

    # Create a bar chart with Plotly Express
    fig = px.bar(
        total_season,
        x='season_name',
        y=['casual', 'registered'],
        barmode='group',
        title='Total bikes rented according to season'
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig)

    avg_temp = df.groupby(['season'])['temp'].mean().reset_index()
    avg_temp['temp'] = avg_temp['temp'] * 41
    avg_temp['season'] = total_season['season_name']

    # Create a line chart with Plotly Express
    fig = px.line(
        avg_temp,
        x='season',
        y='temp',
        title='Average temperature per season'
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig)

    st.write('The graph shows that the demand for bicycles is greatest during the summer, which speaks to the fact that in Washington the average temperature is the highest at that time. It is also evident that the use of the bicycle service is lowest in the winter, when the average temperature is lowest. We do not observe any difference in the seasonality between registered and casual users. With this is mind, we decide that season is better than month granularity for out model, as we would otherwise overfit our model potentially.')


    st.title("Seasonality Features Generation")

    
    
    
    y_casual = df['casual']
    y_regist = df['registered']

    num_lags=24 # to discuss


    st.write('Reviewing Autocorrelation of Casual Riders')
    # Plot the casual time series
    st.line_chart(y_casual)

    # Plot the autocorrelation function of the casual time series
    st.pyplot(plot_acf(y_casual, lags=num_lags))

    # Plot the partial autocorrelation function of the casual time series
    st.pyplot(plot_pacf(y_casual, lags=num_lags, method="ols"))


    st.write('Reviewing Autocorrelation of Registered Riders')
    # Plot the casual time series
    st.line_chart(y_regist)

    # Plot the autocorrelation function of the casual time series
    st.pyplot(plot_acf(y_regist, lags=num_lags))

    # Plot the partial autocorrelation function of the casual time series
    st.pyplot(plot_pacf(y_regist, lags=num_lags, method="ols"))

    st.write('Based on the ACF and PACF plots, key temporal dependencies in the bike-sharing data were identified, leading to the strategic addition of specific features. The plots suggested notable autocorrelations at various lags, prompting the inclusion of lag_1_casual, lag_24_casual, and lag_week_casual to capture short-term, daily, and weekly patterns in usage, respectively. hour_sin and hour_cos effectively encode the cyclical nature of time, a crucial aspect illuminated by the consistent patterns in these plots. The MovingAvg_casual feature, meanwhile, smooths short-term fluctuations, highlighting underlying trends, aligning with the gradual decay observed in the ACF. These features collectively enhance the models ability to understand and predict temporal variations in bike sharing demand')

     # daily seasonality
    df['hour_sin'] = np.sin(2 * np.pi * df['hr'] / 12.0) 
    df['hour_cos'] = np.cos(2 * np.pi * df['hr'] / 12.0)

    # Casual Drivers:
    # moving average of last 7 observations
    df['MovingAvg_casual']=df["casual"].rolling(window=7).mean()

    # Lag 1h
    df['lag_1_casual']=df["casual"].shift(1)
    #Lag 24h
    df['lag_24_casual']=df["casual"].shift(24)

    #weekly seasonality
    df['lag_week_casual']=df["casual"].shift(24*7) 

    # Registered Drivers
    # moving average of last 7 observations
    df['MovingAvg_Reg']=df["registered"].rolling(window=7).mean()

    # Lag 1h
    df['lag_1_Reg']=df["registered"].shift(1)
    #Lag 24h
    df['lag_24_reg']=df["registered"].shift(24)

    #weekly seasonality
    df['lag_week_reg']=df["registered"].shift(24*7) 

    # drop the rows that are NA created through the lagging as the dataset big enough 
    df.dropna(inplace=True) 
    df =df.drop(["dteday","instant","mnth","yr","time_slot"],axis=1)


    # Calculate the correlation matrix
    correlations = df.corr()

    # Display the correlation matrix as a heatmap
    plt.figure(figsize=(15, 15))
    fig, ax = plt.subplots()
    sns.heatmap(correlations, annot=True, cmap='coolwarm', linewidths=0.1)
    plt.subplots_adjust(left=0, right=2, bottom=0, top=2, wspace=0.2, hspace=0.5)

    st.write('Correlation Analysis with new Features for ')
    # Display the heatmap figure
    st.pyplot(fig)

    st.write('Takeaways: We notice that both registered and csual demand are very correlated to their respecitve lags. Temperature also plays a big role, as does time')

    st.title("EDA Conclusion")
    st.write('As a conclusion from the Exploratory Data Analysis (EDA), we can affirm that the demand for bicycle use is influenced by a series of diverse factors, ranging from the day of the week to the temperature and the season of the year in the city. This analysis has allowed us to identify clear patterns in the data that help us better understand user behavior.\n\n'

    'One of the most notable patterns is the difference in bicycle use between registered users and casual users. Registered users show higher demand during weekdays, suggesting that they depend on this mode of transportation for their daily activities. On the other hand, casual users tend to use the service more frequently on weekends, possibly for recreational rides.\n\n'

    'Furthermore, it is relevant to highlight that no missing values were found, so it was not necessary to impute. Some data cleaning tasks were also carried out, such as removing the yr and instant columns, and performing the transformation of the dteday column to a datetime data type to facilitate temporal analysis.\n\n'

    'Another important aspect is the identification of categorical variables that were initially presented as integer data, these variables will be appropriately coded as categories for use in model training, which will ensure accurate representation of this data in analysis and modeling.')





