import pandas as pd
import streamlit as st
import os
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull, save_model as reg_save_model
from pycaret.classification import setup as cls_setup, compare_models as cls_compare_models, pull as cls_pull, save_model as cls_save_model


st.sidebar.image('https://www.inteliment.com/wp-content/uploads/2021/05/Automated-Machine-Learning.png')
st.sidebar.title('Auto Machine Learning')
choice = st.sidebar.radio('Navigation', ['Upload', 'Profiling', 'ML', 'Download'])
st.sidebar.info(
    'This application allows you to build an automated ML Pipeline using Streamlit, Pandas Profiling, and PyCaret')

df = None
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if df is not None:
    if choice == 'Profiling':
        st.title('Automated Exploratory Data Analysis')
        profile_report = ProfileReport(df)
        st_profile_report(profile_report)

    if choice == "ML":
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        chosen_model = st.selectbox('Choose the Model Type' , ['Regression','Classification'])
        if st.button('Run Modelling'):
            with st.spinner('Calculating... Please Wait'):
                if chosen_model == 'Regression':
                    reg_setup(df, target=chosen_target, remove_outliers=True)
                    setup_df = reg_pull()
                    st.dataframe(setup_df)
                    best_model = reg_compare_models()
                    compare_df = reg_pull()
                    st.dataframe(compare_df)
                    reg_save_model(best_model, 'best_model.pkl')
                elif chosen_model == 'Classification':
                    cls_setup(df, target=chosen_target, remove_outliers=True)
                    setup_df = cls_pull()
                    st.dataframe(setup_df)
                    best_model = cls_compare_models()
                    compare_df = cls_pull()
                    st.dataframe(compare_df)
                    cls_save_model(best_model, 'best_model.pkl')
if choice == 'Download':
    if os.path.exists('best_model.pkl'):
        st.download_button('Download Model', 'best_model.pkl', file_name="best_model")
    else:
        st.warning('No trained model found. Please train a model first.')
