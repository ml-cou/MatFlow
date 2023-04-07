import streamlit as st
from sklearn.linear_model import Lasso

def lasso_regression():
    alpha = st.number_input("Alpha", 0.0, 100.0, 1.0, key="ls_alpha")
    fit_intercept = st.checkbox("Fit Intercept", True, key="ls_fit_intercept")
    normalize = st.checkbox("Normalize", False, key="ls_normalize")
    max_iter = st.number_input("Max Iterations", 1, 100000, 1000, key="ls_max_iter")
    selection = st.selectbox("Selection", ["cyclic", "random"], index=0, key="ls_selection")

    try:
        model = Lasso(alpha=alpha, fit_intercept=fit_intercept, normalize=normalize, max_iter=max_iter,
                      selection=selection)
    except ValueError as e:
        # Handle any ValueErrors that occur during initialization
        print("Error: ", e)
    except TypeError as e:
        # Handle any TypeErrors that occur during initialization
        print("Error: ", e)
    except Exception as e:
        # Handle any other exceptions that occur during initialization
        print("Error: ", e)

    return model
