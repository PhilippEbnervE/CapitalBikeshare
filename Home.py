import streamlit as st
from streamlit_option_menu import option_menu
def home_app():

    html_temp = """
    <div style="background-color:lightblue;padding:10px">
    <h2 style="color:white;text-align:center;">Capital Bikeshare Demand Forecasting App</h2>
    </div>"""

    st.markdown(html_temp, unsafe_allow_html=True)

    #st.image('capital-bikeshare-logo.jpg')
    st.markdown('---')
    texto = ('The Bike Demand Forecasting App is a powerful tool designed to provide invaluable assistance to Capital Bikeshare in understanding and predicting future demand. With a user-friendly interface and specialized modules for registered and casual clients, this application offers comprehensive support and data analysis for company management.\n\n'
             'To access any of the available modules and insights, simply click on the left panel of the screen. Then select the name of the module you want, enter the corresponding information (if needed) and finally click on the predict button to obtain the results.')

    # Justificar y alinear el texto
    st.markdown(f'<p style="text-align: justify;">{texto}</p>', unsafe_allow_html=True)