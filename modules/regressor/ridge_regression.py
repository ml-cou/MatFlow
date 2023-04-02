import streamlit as st
from sklearn.linear_model import Ridge

def ridge_regression():
    alpha = st.number_input("Alpha", 0.0, 100.0, 1.0, key="rr_alpha")
    fit_intercept = st.checkbox("Fit Intercept", True, key="rr_fit_intercept")
    normalize = st.checkbox("Normalize", False, key="rr_normalize")
    max_iter = st.number_input("Max Iterations", 1, 100000, 1000, key="rr_max_iter")
    solver = st.selectbox("Solver", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"], index=0, key="rr_solver")

    model = Ridge(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, max_iter=max_iter, solver=solver)

    return model
