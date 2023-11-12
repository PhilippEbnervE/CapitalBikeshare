import joblib
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import casual_app
import registered_app
from datanalysis import data_analysis_app
from casual_app import casual_app
from registered_app import registered_app
from business_case import recommendations_app
from Home import home_app
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title='Home1',
    page_icon='house'
)

class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run():
        # app = st.sidebar(
        with st.sidebar:
            app = option_menu('Select an option',
                              ('Home','Data Analysis','Registered Clients', 'Casual Clients','Recommendations'),
                              icons=['house','bar-chart-line','bicycle','bicycle','bar-chart-line'],
                              menu_icon='check2-circle',
                              styles={
                                  "container": {"padding": "0!important", "background-color": "#fafafa"},
                                  "icon": {"color": "black", "font-size": "15px"},
                                  "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px",
                                               "--hover-color": "#eee"},
                                  "nav-link-selected": {"background-color": "lightblue"},
                              }
                              )
            with st.sidebar:
                st.write('Team members:\n\n'
                         '1. Uxia Couce Reguera.\n\n'
                         '2. Philipp Ebner Von Eschenbach.\n\n'
                         '3. Viviana Florez Camacho.\n\n'
                         '4. Fredrik Jensen.\n\n'
                         '5. Sreesankar Vinu')
                st.image('logoie.png')
        if app == 'Home':
            home_app()
        if app == 'Registered Clients':
            registered_app()
        if app == 'Casual Clients':
            casual_app()

        if app == 'Recommendations':
            recommendations_app()

        elif app == 'Data Analysis':
            data_analysis_app()



    run()

