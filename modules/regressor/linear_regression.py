import streamlit as st
from sklearn.linear_model import LinearRegression

def linear_regression():
    fit_intercept = st.checkbox("Fit Intercept", True, key="lr_fit_intercept")
    normalize = st.checkbox("Normalize", False, key="lr_normalize")
    n_jobs = st.number_input("Number of Jobs", -1, 100, -1, key="lr_n_jobs")

    model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, n_jobs=n_jobs)

    return model
