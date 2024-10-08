import pandas as pd
import streamlit as st
import os
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, pull as reg_pull, save_model as reg_save_model
from pycaret.classification import setup as cls_setup, compare_models as cls_compare_models, pull as cls_pull, save_model as cls_save_model

# Sidebar setup
st.sidebar.image("https://fekrait.com/uploads/topics/16719193475376.jpg")
st.sidebar.title('Auto Machine Learning')
choice = st.sidebar.radio('Navigation', ['Upload', 'Profiling', 'ML', 'Download'])
st.sidebar.info('Build an automated ML Pipeline using Streamlit, ydata_profiling, and PyCaret.')

# Load dataset if it exists
@st.cache_data
def load_data(file):
    return pd.read_csv(file, index_col=None)

# Upload dataset
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset", type=["csv"])
    if file:
        df = load_data(file)
        df.to_csv('dataset.csv', index=None)  
        st.success("Dataset uploaded successfully!")
        st.dataframe(df)

# Data profiling
if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

    if choice == 'Profiling':
        st.title('Automated Exploratory Data Analysis')
        profile_report = ProfileReport(df)
        st_profile_report(profile_report)

    # Machine Learning setup
    if choice == "ML":
        st.title('Machine Learning Modeling')

        # Feature selection
        feature_options = df.columns.tolist()
        selected_features = st.multiselect('Select Features for the Model', feature_options, default=feature_options[:-1])
        chosen_target = st.selectbox('Choose the Target Column', [col for col in feature_options if col not in selected_features])

        # Model type selection
        chosen_model_type = st.selectbox('Choose the Model Type', ['Regression', 'Classification'])
        remove_outliers = st.checkbox('Remove Outliers', value=True)

        if st.button('Run Modeling'):
            with st.spinner('Training the model...'):
                try:
                    # Prepare data for PyCaret
                    if chosen_model_type == 'Regression':
                        reg_setup(df[selected_features + [chosen_target]], target=chosen_target, remove_outliers=remove_outliers)
                        setup_df = reg_pull()
                        st.subheader("Setup Data")
                        st.dataframe(setup_df)

                        best_model = reg_compare_models()
                        compare_df = reg_pull()
                        st.subheader("Model Comparison")
                        st.dataframe(compare_df)

                        # Save the best regression model
                        model_file = 'best_regression_model'
                        reg_save_model(best_model, model_file)
                        st.success(f"Best regression model saved as '{model_file}'.")
                        st.session_state.best_model_type = 'regression'

                    elif chosen_model_type == 'Classification':
                        cls_setup(df[selected_features + [chosen_target]], target=chosen_target, remove_outliers=remove_outliers)
                        setup_df = cls_pull()
                        st.subheader("Setup Data")
                        st.dataframe(setup_df)

                        best_model = cls_compare_models()
                        compare_df = cls_pull()
                        st.subheader("Model Comparison")
                        st.dataframe(compare_df)

                        # Save the best classification model
                        model_file = 'best_classification_model'
                        cls_save_model(best_model, model_file)
                        st.success(f"Best classification model saved as '{model_file}'.")
                        st.session_state.best_model_type = 'classification'

                except Exception as e:
                    st.error(f"An error occurred during modeling: {str(e)}")

# Download best model
if choice == 'Download':
    if hasattr(st.session_state, 'best_model_type'):
        model_file = f'best_{st.session_state.best_model_type}_model'
        if os.path.exists(model_file + '.pkl'):
            with open(model_file + '.pkl', 'rb') as f:
                st.download_button('Download Best Model', f, file_name=f"{model_file}.pkl")
        else:
            st.warning('No trained model found. Please train a model first.')
    else:
        st.warning('No trained model found. Please train a model first.')
