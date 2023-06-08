import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


def hyperparameter_optimization(X_train, y_train):
    
    st.subheader("Hyperparameter Optimization Settings")
    cv = st.number_input("Number of cross-validation folds", min_value=2, value=5, step=1)
    random_state = st.number_input("Random state for hyperparameter search", min_value=0, value=0, step=1)

    if "lr_best_param" not in st.session_state:
        st.session_state.lr_best_param = {
            "fit_intercept": True
        }

    st.write('#')

    if st.button('Run Optimization'):

        st.write('#')
        param_grid = {
            "fit_intercept": [True, False],
        }
        model = LinearRegression()

        with st.spinner('Doing hyperparameter optimization...'):

            clf = GridSearchCV(model, param_grid=param_grid, cv=cv, n_jobs=-1)
            st.spinner("Fitting the model...")
            clf.fit(X_train, y_train)

        st.success("Hyperparameter optimization completed!")

        best_params = clf.best_params_

        st.write('Best estimator:')

        rows = []
        for param in best_params:
            rows.append([param, best_params[param]])
        st.table(pd.DataFrame(rows, columns=['Parameter', 'Value']))

        st.session_state.opti_results_df = pd.DataFrame(rows, columns=['Parameter', 'Value'])

        st.session_state.lr_best_param = best_params

    return st.session_state.lr_best_param

def linear_regression(X_train, y_train):
    best_params = hyperparameter_optimization(X_train, y_train)

    st.subheader("Model Settings")

    fit_intercept = st.checkbox(
        "Fit Intercept", key="lr_fit_intercept1",
        value= best_params['fit_intercept']
    )
    n_jobs = st.number_input("Number of Jobs", -1, 100, -1, key="lr_n_jobs")

    try:
        model = LinearRegression(
            fit_intercept=fit_intercept,  n_jobs=n_jobs
        )
    except ValueError as e:
        st.error(f"Error: {e}")
        model = None

    return model
